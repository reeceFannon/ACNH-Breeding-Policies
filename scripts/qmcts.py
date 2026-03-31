import math
import random
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Sequence, FrozenSet
from transitions import FlowerTransitions, TransitionTensorBuilder, TransitionTensor, posterior_given_phenotype, canonical_pair
from qmdp import QuantumFlowerMDP, QuantumState, QuantumAction, QuantumFlower, create_flower_hash, quantum_pair
from optimize import BreedingPolicyNet, optimize_policy, policy_grad

def initialize_start(genotypes: List[str], transition_tensor: TransitionTensor) -> QuantumState:
  p = len(genotypes)
  idxs = [transition_tensor.genotype_to_idx[genotype] for genotype in genotypes]
  phenotypes = [transition_tensor.idx_to_phenotype[idx] for idx in idxs]
  hashes = [create_flower_hash(phenotype, ("0"*16, "0"*16)) for phenotype in phenotypes]
  flowers = [QuantumFlower(hashes[i], phenotypes[i], True, (idxs[i],), (1.0,), ("0"*16, "0"*16)) for i in range(p)]
  return frozenset(set(flowers))

def resolve_latent_genotype(flower: QuantumFlower, transition_tensor: TransitionTensor, cache: Dict[QuantumFlower, int]) -> int:
  if flower not in cache:
    dist = flower.to_dense(len(transition_tensor.idx_to_genotype), device = transition_tensor.T.device, dtype = transition_tensor.T.dtype)
    cache[flower] = int(torch.multinomial(dist, 1).item())
  return cache[flower]

def sample_child(action: QuantumAction, transition_tensor: TransitionTensor, cache: Dict[QuantumFlower, int]) -> tuple[QuantumFlower, int]:
  parent1, parent2 = quantum_pair(action)
  i = resolve_latent_genotype(parent1, transition_tensor, cache)
  j = resolve_latent_genotype(parent2, transition_tensor, cache)

  T = transition_tensor.T
  offspring_dist = T[i, j, :]
  child_idx = int(torch.multinomial(offspring_dist, 1).item())
  child_phenotype = transition_tensor.idx_to_phenotype[child_idx]
  
  posterior = posterior_given_phenotype(offspring_dist, child_idx, transition_tensor)
  parent_labels = canonical_pair(parent1.hash, parent2.hash)
  child = QuantumFlower.from_distribution(posterior, phenotype = child_phenotype, parents = parent_labels)

  return child, child_idx

def sample_next_state(state: QuantumState, action: QuantumAction, transition_tensor: TransitionTensor, cache: Dict[QuantumFlower, int]) -> QuantumState:
  child, child_idx = sample_child(action, transition_tensor, cache)
  cache[child] = child_idx
  next_state = frozenset(set(state) | {child})
  return next_state

def action_heuristic_score(action: QuantumAction, gradients: torch.FloatTensor, N: int) -> float:
  parent1, parent2 = quantum_pair(action)
  d1 = parent1.to_dense(N, device = gradients.device, dtype = gradients.dtype)
  d2 = parent2.to_dense(N, device = gradients.device, dtype = gradients.dtype)
  return float(torch.dot(d2, gradients@d1).item())

def random_rollout(qmdp: QuantumFlowerMDP, start_state: QuantumState, max_depth: int = 20, heuristic: bool = False, transition_tensor: TransitionTensor = None, gradients: torch.FloatTensor = None) -> float:
  """
  Random policy rollout from start_state.
  Returns reward = -steps to terminal (more negative if slower).
  """
  latent_genotype_cache: Dict[QuantumFlower, int] = {}
  prev_state = frozenset()
  state = start_state
  N = len(transition_tensor.idx_to_genotype)
    
  for t in range(max_depth):
    if qmdp.is_terminal(state): return float(-t) # already reached target

    actions = qmdp.available_actions(state)
    if not actions: return float(-max_depth) # dead-end; can't continue

    if heuristic:
      if state - prev_state: # if the set difference exists
        D1 = [a.to_dense(N, device = gradients.device, dtype = gradients.dtype) for a, _ in actions]
        D2 = [b.to_dense(N, device = gradients.device, dtype = gradients.dtype) for _, b in actions]
        scores = torch.tensor([torch.dot(d2, gradients@d1) for d1, d2 in zip(D1, D2)])
        bias = torch.softmax(scores, dim = -1)
      action = random.choices(actions, weights = bias, k = 1)[0]
    else:
      action = random.choice(actions)

    prev_state = state
    state = sample_next_state(state, action, transition_tensor, latent_genotype_cache)

  return float(-max_depth) # didn't reach terminal within horizon

@dataclass
class MCTSNode:
  state: QuantumState
  parent: Optional["MCTSNode"] = None
  action_from_parent: Optional[QuantumAction] = None
  
  children: Dict[QuantumAction, "MCTSNode"] = field(default_factory = dict)
  untried_actions: List[QuantumAction] = field(default_factory = list)

  visits: int = 0
  total_reward: float = 0.0
  total_sq_reward: float = 0.0

  action_score_cache: Optional[Dict[QuantumAction, float]] = None

  def is_fully_expanded(self) -> bool:
    return len(self.untried_actions) == 0

  def widening_limit(self, c_pw: float, alpha_pw: float) -> int:
    return max(1, int(c_pw*(self.visits**alpha_pw)))

  def can_expand(self, c_pw: float, alpha_pw: float) -> bool:
    return (len(self.untried_actions) > 0 and len(self.children) < self.widening_limit(c_pw, alpha_pw))

  def update_action_score_cache(self, gradients: Optional[torch.Tensor], N: int) -> None:
    if self.action_score_cache is not None and self.sorted_untried_actions is not None: return
    self.action_score_cache = {action: action_heuristic_score(action, dL_max, N) for action in self.untried_actions}
    self.untried_actions = sorted(self.untried_actions, key=lambda a: self.action_score_cache[a], reverse=True)

  def best_child(self, c: float = math.sqrt(2.0)) -> "MCTSNode":
    """
    UCT selection: argmax over children of (Q / N + c * sqrt(log Np / Nc)).
    """
    best_score = -float("inf")
    best = None
    for child in self.children.values():
      if child.visits == 0:
        uct = float("inf")
      else:
        exploit = child.total_reward / child.visits
        explore = c*math.sqrt(math.log(self.visits)/child.visits)
        uct = exploit + explore
      if uct > best_score:
        best_score = uct
        best = child
    return best

def mcts_search(qmdp: QuantumFlowerMDP, root_state: QuantumState, n_simulations: int = 1000, max_rollout_depth: int = 20, c: float = math.sqrt(2.0), heuristic: bool = False, cloning: bool = False, transition_tensor: TransitionTensor = None, gradients: Dict = None, init_logits_scale: float = 0.01, optim_steps: int = 1000, lr: float = 1e-2, log_steps: int = 100, eps_present: float = 0.0, c_pw: float = 1.0, a_pw: float = 0.5) -> MCTSNode:
  """
  Run MCTS from the root_state and return the root node
  with its tree of children filled in.
  """
  root = MCTSNode(state = root_state, parent = None, action_from_parent = None, untried_actions = qmdp.available_actions(root_state))

  dl_max = None
  N = len(transition_tensor.idx_to_genotype)
  if heuristic:
    dL = gradients["grad_logits"]
    mu = dL.mean(dim = 1, keepdim = True)
    sigma = dL.std(dim = 1, keepdim = True)
    dL_Z = (dL - mu)/sigma
    dL_max = torch.maximum(dL_Z, dL_Z.T)

  for _ in range(n_simulations):
    node = root

    # 1. SELECTION: descend tree using UCT
    while not qmdp.is_terminal(node.state):
      if node.can_expand(c_pw, alpha_pw): break
      if node.children: node = node.best_child(c)
      else: break

    # 2. EXPANSION: expand one untried action (if any and not terminal)
    if not qmdp.is_terminal(node.state) and node.can_expand(c_pw, alpha_pw):
      node.update_action_score_cache(dL_max, N)
      action = node.untried_actions[0]
      node.untried_actions.remove(action)

      next_state, _ = qmdp.sample_next_state(node.state, action)

      child = MCTSNode(state = next_state, parent = node, action_from_parent = action, untried_actions = qmdp.available_actions(next_state))
      node.children[action] = child
      node = child

    # 3. SIMULATION (ROLLOUT): from this leaf node
    reward = random_rollout(qmdp, node.state, max_depth = max_rollout_depth, heuristic = heuristic, transition_tensor = transition_tensor, gradients = dL_max)

    # 4. BACKPROPAGATION
    while node is not None:
      node.visits += 1
      node.total_reward += reward
      node.total_sq_reward += reward**2
      node = node.parent

  return root

def full_episode(species: str, targets: List[str], root_state: List[str], root_counts: List[float], root_n_simulations: int = 1000, max_episode_steps: int = 1000, root_max_rollout_depth: int = 20, c: float = math.sqrt(2.0), min_n_simulations: int = 100, max_simulations_scale_factor: float = 0.0, min_depth_floor: int = 10, seed: int | None = None, heuristic: bool = False, cloning: bool = False, num_waves: int = 4, init_logits_scale: float = 0.01, optim_steps: int = 1000, lr: float = 1e-2, log_steps: int = 100, eps_present: float = 0.0, recalc_heuristic_every: int = 1) -> Dict[str, object]:
  total_steps = 0
  trajectory: List[Tuple[QuantumState, QuantumAction]] = []
  current_max_rollout_depth = root_max_rollout_depth
  current_n_simulations = root_n_simulations
  step_stats = StepStats()
  success = False
  episode_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
  random.seed(episode_seed)
  transition_tensor = TransitionTensorBuilder().build_transition_tensor(species)
  x = torch.zeros(len(transition_tensor.idx_to_genotype), device = transition_tensor.T.device)
  state = initialize_start(root_state, transition_tensor)

  N = len(transition_tensor.idx_to_genotype)
  for flower, count in zip(state, root_counts): x += count*flower.to_dense(N, device = x.device, dtype = x.dtype)
  target_idx = []
  for pheno in targets: target_idx.extend(transition_tensor.phenotype_to_idx[pheno])
  target_idx = torch.tensor(target_idx, device = transition_tensor.T.device)
  
  if not heuristic: gradients = None
    
  while (not success) and (total_steps < max_episode_steps):
    if current_max_rollout_depth <= 0: break

    if (heuristic) and (total_steps % recalc_heuristic_every == 0):
      model = BreedingPolicyNet(transition_tensor, num_waves, cloning = cloning, init_logits_scale = init_logits_scale)
      optimize_policy(model, x, target_idx, steps = optim_steps, lr = lr, log_steps = log_steps)
      gradients = policy_grad(model, x, target_idx, eps_present = eps_present)

    # --- MCTS from current state ---
    qmdp = QuantumFlowerMDP(species = species, transition_tensor = transition_tensor, targets = targets)
    root = mcts_search(qmdp, state, current_n_simulations, current_max_rollout_depth, c, heuristic, cloning, transition_tensor, gradients, init_logits_scale, optim_steps, lr, log_steps, eps_present)
    root_stats = extract_root_action_stats(root)
    best_action = best_root_action_from_stats(root_stats)

    if best_action is None: break  # no legal actions

    # sample next state in the *main* process
    next_state, child = qmdp.sample_next_state(state, best_action)
    state = next_state
    x += child.to_dense(N, device = x.device, dtype = x.dtype)
    total_steps += 1
    parent1, parent2 = best_action
    print(f"Finished step {total_steps}: Chose action ({parent1.phenotype}, {parent2.phenotype}) and produced offspring ({child.phenotype}) with {current_n_simulations} rollouts at a depth of {current_max_rollout_depth}")

    trajectory.append((state, best_action, child))

    # update best action stats and rollout depth
    best_stats = root_stats[best_action]
    step_stats.update(best_stats["visits"], -best_stats["total_reward"], best_stats["total_sq_reward"])
    mu, sigma = step_stats.mean, math.sqrt(max(step_stats.variance, 0.0))
    current_max_rollout_depth = int(max(min(mu + 3*sigma, current_max_rollout_depth), min_depth_floor))

    # update the number of simulations
    visits = [stats["visits"] for stats in root_stats.values()]
    if len(visits) == 0:
      entropy_ratio = 0.0
    else:
      p_visit = [v/sum(visits) for v in visits]
      entropy = -sum(p_v*math.log(p_v) for p_v in p_visit)
      entropy_ratio = entropy/math.log(len(visits)) # the upper bound on entropy is ln(len(X)), where every x in X is identically equal (no uncertainty)
    scale_factor = max(entropy_ratio, max_simulations_scale_factor)
    current_n_simulations = int(max(scale_factor*current_n_simulations, min_n_simulations)) # update simulations in accordance with uncertainty/precision about actions
    
    success = qmdp.is_terminal(state)

  return {"species": species, "targets": targets, "trajectory": trajectory, "final_state": state, "total_steps": total_steps, "success": success}


@dataclass
class StepStats:
  n: int = 0
  sum_steps: float = 0.0
  sum_sq_steps: float = 0.0

  def update(self, n: int, sum_steps: float, sum_sq_steps: float) -> None:
    self.n = n
    self.sum_steps = sum_steps
    self.sum_sq_steps = sum_sq_steps

  @property
  def mean(self) -> Optional[float]:
    if self.n == 0:
      return None
    return self.sum_steps/self.n

  @property
  def variance(self) -> Optional[float]:
    if self.n == 0:
      return None
    m = self.mean
    return (self.sum_sq_steps/self.n) - m**2 # E(X^2) - (E(X))^2

def extract_root_action_stats(root: MCTSNode) -> Dict[QuantumAction, Dict[str, float]]:
  """
  Summarize stats for each root action:
  returns {action: {"visits": v, "total_reward": r}}.
  """
  stats: Dict[QuantumAction, Dict[str, float]] = {}
  for action, child in root.children.items():
    stats[action] = {"visits": child.visits,
                      "total_reward": child.total_reward,
                      "total_sq_reward": child.total_sq_reward}
  return stats

def best_root_action_from_stats(stats: Dict[QuantumAction, Dict[str, float]]) -> Optional[QuantumAction]:
  if not stats:
    return None
  return max(stats.items(), key=lambda kv: kv[1]["visits"])[0]
