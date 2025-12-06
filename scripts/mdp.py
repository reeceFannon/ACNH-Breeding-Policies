import random
from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Tuple, Dict, Iterable, List, Sequence, Optional
from itertools import combinations_with_replacement
from transitions import FlowerTransitions, canonical_pair
from testing import (ambiguous_groups_from_cross, choose_optimal_tester_for_group, TestSpec, disjoint_sets_for_tester)

TEST_ACTION = ("__TEST__",)

@dataclass(frozen = True)
class KnowledgeState:
  known: FrozenSet[str]
  pending: Optional[TestSpec] = None

State = KnowledgeState
Action = Tuple[str, ...]

@dataclass
class FlowerMDP:
  species: str
  transitions: FlowerTransitions
  genetics: FlowerGenetics
  targets: Sequence[FrozenSet[str]]
  action_cache: Dict[FrozenSet[str], List[Action]] = field(default_factory = dict, init = False, repr = False)

  def is_terminal(self, state: State) -> bool:
    """
    Terminal if for every target group G in target_groups,
    state ∩ G ≠ ∅ (i.e., we have at least one genotype from each group).
    """
    known = state.known
    for group in self.targets:
      if known.isdisjoint(group):
        return False
    return True

  def available_actions(self, state: State) -> List[Action]:
    """
    All unordered parent pairs (A,B) with A,B in state.
    - Normal breeding actions use only known genotypes.
    - If state.pending is not None, we also add a TEST action.
    """
    known = state.known
    actions: List[Action] = []

    if known:
      if known in self._action_cache:
        actions = list(self._action_cache[known])
        else:
          genos = sorted(known)
          actions = list(combinations_with_replacement(genos, 2))
          self._action_cache[known] = list(actions)

    if state.pending is not None:
      actions = list(actions)
      actions.append(TEST_ACTION)

    return actions

  def is_globally_ambiguous_phenotype(self, phenotype: str) -> bool:
    """
    True if this phenotype corresponds to > 1 genotype for this species
    (using global genetics table).
    """
    genos = self.genetics.genotypes_for_phenotypes(self.species, (phenotype,))[0] # first (and only) group
    return len(genos) > 1

  def next_state_test(self, state: State) -> Dict[State, float]:
    pending = state.pending
    if pending is None:
      return {state: 1.0}

    tester = pending.tester
    group = list(pending.group) # [(h, P(h|ambiguous)), ...]
    candidates = [g for g, _ in group]

    # 1) For tester and candidate set, compute unique phenotypes per genotype
    unique_pheno, _hit_prob = disjoint_sets_for_tester(self.species, tester, candidates, self.transitions, self.genetics)

    # 2) Enumerate outcomes: for each true h and each offspring child from tester×h
    dist: Dict[State, float] = {}
    for h, p_h in group:
      offspring = self.transitions.offspring_distribution(self.species, tester, h)
      if not offspring:
        # no child; this test attempt gives no info
        dist[state] = dist.get(state, 0.0) + p_h
        continue

      for child_g, p_child in offspring.items():
        ph_child = self.genetics.phenotype_for_genotype(self.species, child_g)

        # total probability of this outcome: P(h) * P(child | h, tester)
        p_outcome = p_h*p_child

        if ph_child is None:
          # unrecognized phenotype – treat as non-informative
          next_state = state
        else:
          # success if child's phenotype is unique to this h
          if ph_child in unique_pheno.get(h, set()):
            # We learn h's genotype.
            new_known = set(state.known)
            new_known.add(h)
            next_state = KnowledgeState(frozenset(new_known), pending = None)
          else:
            # phenotype does not uniquely identify h -> no info
            # per your rule: do nothing about offspring; same state.
            next_state = state

        dist[next_state] = dist.get(next_state, 0.0) + p_outcome

    # Normalize to be safe
    total = sum(dist.values())
    if total > 0:
      for k in list(dist.keys()):
        dist[k] /= total

    return dist

  def next_state_distribution(self, state: State, action: Action) -> Dict[State, float]:
    """
    Given a state S and action (A,B), return a distribution over next states.
    Each offspring genotype g is added to the set S (if not already present).
    """
    if action == TEST_ACTION:
      return self.next_state_test(state)
      
    a, b = canonical_pair(*action)
    offspring_dist = self.transitions.offspring_distribution(self.species, a, b)

    if not offspring_dist:
      return {state: 1.0}

    dist: Dict[State, float] = {}
    ambiguous = ambiguous_groups_from_cross(self.species, a, b, self.transitions, self.genetics)
    # Map genotype -> (phenotype, group[(geno,prob),...]) if ambiguous
    geno_to_group: Dict[str, Tuple[str, List[Tuple[str, float]]]] = {}
    for ph, group in ambiguous.items():
      for g, prob in group:
        geno_to_group[g] = (ph, group)

    for g, p in offspring_dist.items():
      # Determine phenotype
      ph = self.genetics.phenotype_for_genotype(self.species, [g])

      # Start from baseline known set
      known = set(state.known)
      pending = state.pending

      if ph is None:
        # Unknown phenotype: ignore, no unlock
        next_state = KnowledgeState(frozenset(known), pending)
      else:
        # Is this genotype part of an ambiguous phenotype for this cross?
        if g in geno_to_group:
          ph_ambig, group = geno_to_group[g]

          # Compute optimal tester given current known genotypes.
          test_spec = choose_optimal_tester_for_group(self.species, state.known, ph_ambig, group, self.transitions, self.genetics)

          # If we found a useful test, install it as pending.
          # Otherwise, we effectively learn nothing (we treat it as unusable).
          if test_spec is not None:
            pending = test_spec
          # known unchanged
          next_state = KnowledgeState(frozenset(known), pending)

        else:
          # Phenotype is unambiguous in this cross; we can safely add g.
          known.add(g)
          next_state = KnowledgeState(frozenset(known), pending)

    dist[next_state] = dist.get(next_state, 0.0) + prob

  return dist
