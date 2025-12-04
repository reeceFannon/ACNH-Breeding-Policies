from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Tuple, Dict, Iterable, List
import random
from itertools import combinations_with_replacement
from transitions import FlowerTransitions, canonical_pair

State = FrozenSet[str]
Action = Tuple[str, str]

@dataclass
class FlowerMDP:
  species: str
  transitions: FlowerTransitions
  targets: FrozenSet[str]
  action_cache: Dict[State, List[Action]] = field(default_factory = dict, init = False, repr = False)

  def is_terminal(self, state: State) -> bool:
    """Terminal if any target genotype is present in the unlocked set."""
    return not self.targets.isdisjoint(state)

  def available_actions(self, state: State) -> List[Action]:
    """
    All unordered parent pairs (A,B) with A,B in state.
    You can later prune / prioritize this if needed.
    """
    if state in self.action_cache:
      return self.action_cache[state]

    genotypes = sorted(state)
    actions = list(combinations_with_replacement(genotypes, 2))
    return actions

  def next_state_distribution(self, state: State, action: Action) -> Dict[State, float]:
    """
    Given a state S and action (A,B), return a distribution over next states.
    Each offspring genotype g is added to the set S (if not already present).
    """
    a, b = canonical_pair(*action)
    offspring_probs = self.transitions.offspring_distribution(self.species, a, b)

    if not offspring_probs:
      return {state: 1.0}

    dist: Dict[State, float] = {}
    for g, p in offspring_probs.items():
      next_state = frozenset(set(state) | {g})
      dist[next_state] = dist.get(next_state, 0.0) + p

    return dist
