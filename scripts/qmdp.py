from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Tuple, Dict, Iterable, List, Sequence, Optional
import random
from itertools import combinations_with_replacement
import torch

from transitions import (FlowerTransitions, TransitionTensor, canonical_pair, posterior_given_phenotype)

QuantumState = FrozenSet["QuantumFlower"]
QuantumAction = Tuple["QuantumFlower", "QuantumFlower"]

@dataclass(frozen = True)
class QuantumFlower:
  phenotype: str
  genotype_idxs: Tuple[int, ...]
  genotype_probs: Tuple[float, ...]
  parents: Optional[Tuple[str, str]] = None

  def to_dense(self, N: int, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert sparse support/probabilities into a dense [N] probability vector.
    """
    out = torch.zeros(N, device = device, dtype = dtype)
    idx = torch.tensor(self.genotype_idxs, device = device, dtype = torch.long)
    probs = torch.tensor(self.genotype_probs, device = device, dtype = dtype)
    out[idx] = probs
    return out

  @staticmethod
  def from_distribution(dist: torch.Tensor, phenotype: str, *, parents: Optional[Tuple[str, str]] = None, eps: float = 1e-12, decimals: int = 12) -> "QuantumFlower":
    """
    Build a QuantumFlower from a dense genotype distribution by dropping zero-mass entries
    and storing the remainder sparsely.
    `decimals` is used only to stabilize hashing / equality against tiny float noise.
    """
    idxs = torch.nonzero(dist > eps, as_tuple = False).flatten()
    probs = dist[idxs]
    probs = probs/probs.sum()

    genotype_idxs = tuple(int(i) for i in idxs.tolist())
    genotype_probs = tuple(round(float(p), decimals) for p in probs.tolist())

    return QuantumFlower(phenotype = phenotype, genotype_idxs = genotype_idxs, genotype_probs = genotype_probs, parents = parents)

@dataclass
class QuantumFlowerMDP:
  species: str
  transition_tensor: TransitionTensor
  targets: Sequence[FrozenSet[str]]
  action_cache: Dict[QuantumState, List[QuantumAction]] = field(default_factory = dict, init = False, repr = False)

  def is_terminal(self, state: BeliefState) -> bool:
    """
    Terminal if, for every target phenotype group, the state contains at least one
    flower whose observed phenotype belongs to that group.
    """
    phenotypes = {flower.phenotype for flower in state}
    for group in self.targets:
      if phenotypes.isdisjoint(group):
        return False
    return True

  def available_actions(self, state: QuantumState) -> List[QuantumAction]:
    """
    All unordered breeding pairs from the current Quantum flowers.
    """
    if state in self.action_cache: return self.action_cache[state]

    flowers = sorted(state, key = lambda f: (f.phenotype, f.genotype_idxs, f.genotype_probs, f.parents or ("", "")))
    actions = list(combinations_with_replacement(flowers, 2))
    self.action_cache[state] = actions
    return actions

  def breed_distribution(self, flower1: QuantumFlower, flower2: QuantumFlower) -> torch.Tensor:
    """
    Compute offspring genotype distribution from two ambiguous parents:
        offspring[k] = sum_{a,b} d1[a] d2[b] T[a,b,k]
    Returns a dense probability vector of length N.
    """
    T = self.transition_tensor.T
    N = T.shape[0]

    d1 = flower1.to_dense(N = N, device = T.device, dtype = T.dtype) # parent 1 distribution vector
    d2 = flower2.to_dense(N = N, device = T.device, dtype = T.dtype) # parent 2 distribution vector
    offspring = torch.einsum("a,b,abk->k", d1, d2, T)                # offspring distribution vector

    total = offspring.sum()
    if total.item() <= 0: return offspring

    return offspring/total

  def sample_offspring_flower(self, flower1: QuantumFlower, flower2: QuantumFlower, *, eps: float = 1e-12) -> QuantumFlower:
    """
    Sample a latent offspring genotype from the mixed offspring distribution,
    observe only its phenotype, then return the phenotype-conditioned posterior flower.
    """
    offspring_dist = self.breed_distribution(flower1, flower2)
    idx = int(torch.multinomial(offspring_dist, 1).item())
    phenotype = self.transition_tensor.idx_to_phenotype[idx]
    posterior = posterior_given_phenotype(offspring_dist, idx, self.transition_tensor)

    parent_labels = (flower1.phenotype, flower2.phenotype)
    return QuantumFlower.from_distribution(posterior, phenotype = phenotype, parents = canonical_pair(*parent_labels))

def sample_next_state(self, state: QuantumState, action: QuantumAction, *) -> tuple[QuantumState, QuantumFlower]:
    """
    Sample one offspring flower and add it to the state.
    """
    flower1, flower2 = action
    child = self.sample_offspring_flower(flower1, flower2)
    next_state = frozenset(set(state) | {child})
    return next_state, child
