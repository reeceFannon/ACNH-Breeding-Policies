from typing import List, Dict, Any
import torch
import torch.nn as nn

def parent_indices_by_wave(waves_out: List[Dict[str, Any]], geno_to_idx: Dict[str, int], *, device = None) -> List[torch.LongTensor]:
  Ei: List[torch.LongTensor] = []
  for w in waves_out:
    parent1 = torch.tensor([geno_to_idx[a["parent1"]] for a in w["actions"]], dtype = torch.long, device = device)
    parent2 = torch.tensor([geno_to_idx[a["parent2"]] for a in w["actions"]], dtype = torch.long, device = device)
    parentU = torch.unique(torch.cat([parent1, parent2], dim = 0), sorted = True)
    Ei.append(parentU)
  return Ei

def build_Ti_list(T: torch.Tensor, wave_parent_idx: List[torch.LongTensor]) -> List[torch.Tensor]:
  Ti = []
  for Ei in wave_parent_idx: Ti.append(T.index_select(0, Ei))  # [P, N, N]
  return Ti

def masked_row_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
  neg_large = torch.finfo(logits.dtype).min
  masked_logits = torch.where(mask, logits, torch.tensor(neg_large, device = logits.device, dtype = logits.dtype))
  probs = torch.softmax(masked_logits, dim = dim)

  allowed_count = mask.sum(dim = dim, keepdim = True)
  probs = torch.where(allowed_count > 0, probs, torch.zeros_like(probs))
  return probs

class BreedingPolicyNet(nn.Module):
  """
  Wave i:
    - fixed parent1 pool indices Ei (length P_i)
    - fixed Ti = T[Ei,:,:] with shape [P_i, N, N]
    - learn logits_i of shape [P_i, N] => Q_i row-softmax over global parent2

  Residual update:
    x <- x + offspring
  """
  def __init__(self, Ti_list: List[torch.Tensor], Ei_list: List[torch.LongTensor], *, init_logits_scale: float = 0.01):
    super().__init__()
    assert len(Ti_list) == len(Ei_list)
    self.num_waves = len(Ti_list)
    N = int(Ti_list[0].shape[1])
    self.register_buffer("_N", torch.tensor(N, dtype = torch.long))

    self._Ti_names = []
    self._Ei_names = []

    for i, (Ti, Ei) in enumerate(zip(Ti_list, Ei_list)):
      if Ti.shape[1] != N or Ti.shape[2] != N:
        raise ValueError(f"All Ti must have shape [P, N, N] with same N. Wave {i} has {tuple(Ti.shape)}")
      self.register_buffer(f"Ti_{i}", Ti)
      self.register_buffer(f"Ei_{i}", Ei)
      self._Ti_names.append(f"Ti_{i}")
      self._Ei_names.append(f"Ei_{i}")

    # Learnable logits per wave: [P_i, N]
    self.logits = nn.ParameterList([nn.Parameter(init_logits_scale*torch.randn(int(Ei.numel()), N)) for Ei in Ei_list])
    self.Q: List[torch.Tensor | None] = [None]*self.num_waves

  @property
  def N(self) -> int:
    return int(self._N.item())

  def forward(self, x0: torch.Tensor, target_idx: torch.LongTensor, *, eps_present: float = 0.0, save_Q: bool = False) -> torch.Tensor:
    """
    x0: [N] expected counts
    eps_present: treat x > eps_present as present for masking
    """
    if x0.ndim != 1 or x0.shape[0] != self.N:
      raise ValueError(f"x0 must be shape [N={self.N}], got {tuple(x0.shape)}")

    x = torch.clamp(x0, min=0.0)

    for i in range(self.num_waves):
      Ti = getattr(self, self._Ti_names[i])     # [P, N, N]
      Ei = getattr(self, self._Ei_names[i])     # [P]
      logits = self.logits[i]                   # [P, N]

      x_p1 = x.index_select(0, Ei)              # [P]
      present_p1 = (x_p1 > eps_present)         # [P]
      present_p2 = (x > eps_present)            # [N]
      present_p2[target_idx] = False            # Ban targets as parents

      # Allowed parent2 choices depend on parent2 availability; rows also depend on parent1 availability
      allowed = present_p1[:, None] & present_p2[None, :]   # [P, N]
      Q = masked_row_softmax(logits, allowed, dim = 1)      # [P, N]

      if save_Q: self.Q[i] = Q.detach() # remove autograd graph

      # offspring[k] = sum_a x_p1[a] * sum_b Q[a,b] * Ti[a,b,k]
      offspring = torch.einsum("a,ab,abk->k", x_p1, Q, Ti)    # [N]
      clones = torch.zeros_like(x)
      clones[target_idx] = x[target_idx]

      x = torch.clamp(x + offspring + clones, min = 0.0)

    return x

def optimize_policy(model: BreedingPolicyNet, x0: torch.Tensor, target_idx: torch.LongTensor, *, steps: int = 2000, lr: float = 1e-2, eps_present: float = 1e-6, clip_grad: float = 1.0, log_steps: int = 50):
  opt = torch.optim.Adam(model.parameters(), lr = lr)
  for t in range(steps):
    opt.zero_grad()

    x_final = model(x0, target_idx, eps_present = eps_present, save_Q = False)
    target_mass = x_final.index_select(0, target_idx).sum()
    loss = -target_mass
    loss.backward()

    if clip_grad is not None:
      torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    opt.step()

    if (t % log_steps) == 0:
      with torch.no_grad():
        metric = target_mass.item()
      print(f"step {t:4d} | loss {loss.item():.6g} | metric {metric:.6g}")

  # One final pass to cache Q_i using the final parameters
  model.eval()
  with torch.no_grad():
    _ = model(x0, target_idx, eps_present = eps_present, save_Q = True) # only save once at end to avoid overhead
  
  return model
