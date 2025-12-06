from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from transitions import FlowerTransitions, FlowerGenetics

def ambiguous_groups_from_cross(species: str, parent1: str, parent2: str, transitions: FlowerTransitions, genetics: FlowerGenetics):
  """
  From A×B, return a dict:
      phenotype -> [(genotype, prob), ...]
  only for phenotypes that correspond to >1 genotype (ambiguous).
  """
  G = transitions.offspring_distribution(species, parent1, parent2)
  genotypes = list(G.keys())
  probs = [G[g] for g in genotypes]
  phenotypes = genetics.phenotypes_for_genotypes(species, genotypes)

  by_pheno: dict[str, list[tuple[str, float]]] = defaultdict(list)
  for g, p, ph in zip(genotypes, probs, phenotypes):
    by_pheno[ph].append((g, p))

  # keep only ambiguous phenotypes
  ambiguous = {ph: lst for ph, lst in by_pheno.items() if len(lst) > 1}
  return ambiguous  # phenotype -> [(geno, prob), ...]

def phenotype_distribution_for_pair(species: str, parent1: str, parent2: str, transitions: FlowerTransitions, genetics: FlowerGenetics) -> dict[str, float]:
  """
  offspring phenotype -> probability for (parent1, parent2).
  """
  geno_dist = transitions.offspring_distribution(species, parent1, parent2)
  pheno_dist: dict[str, float] = {}
  if not geno_dist:
    return pheno_dist

  genos = list(geno_dist.keys())
  probs = [geno_dist[g] for g in genos]
  phenos = genetics.phenotypes_for_genotypes(species, genos)

  for g, p, ph in zip(genos, probs, phenos):
    pheno_dist[ph] = pheno_dist.get(ph, 0.0) + p

  return pheno_dist

def disjoint_sets_for_tester(species: str, tester: str, candidates: list[str], transitions: FlowerTransitions, genetics: FlowerGenetics):
  """
  For a given tester s and ambiguous group C = candidates:
    - For each h in C, compute phenotype distribution for s×h
    - Find phenotypes that are unique to each h
    - Return:
        unique_pheno[h] = set of phenotypes that identify h
        hit_prob[h] = probability of seeing a phenotype in unique_pheno[h]
  """
  # 1) phenotype distributions per candidate
  pheno_by_h: dict[str, dict[str, float]] = {}
  all_phenos: set[str] = set()

  for h in candidates:
    phi = phenotype_distribution_for_pair(species, tester, h, transitions, genetics)
    pheno_by_h[h] = phi
    all_phenos.update(phi.keys())

  # 2) which genotypes can produce each phenotype?
  producers: dict[str, set[str]] = {p: set() for p in all_phenos}
  for h, phi in pheno_by_h.items():
    for p, prob in phi.items():
      if prob > 0:
        producers[p].add(h)

  # 3) phenotypes uniquely identifying each genotype
  unique_pheno: dict[str, set[str]] = {h: set() for h in candidates}
  hit_prob: dict[str, float] = {h: 0.0 for h in candidates}

  for p, srcs in producers.items():
    if len(srcs) == 1:
      (only_h,) = tuple(srcs)
      unique_pheno[only_h].add(p)
      # add probability of p when true genotype = only_h
      hit_prob[only_h] += pheno_by_h[only_h].get(p, 0.0)

  return unique_pheno, hit_prob

def tester_reveal_probability(species: str, tester: str, group: list[tuple[str, float]], transitions: FlowerTransitions, genetics: FlowerGenetics):
  """
  group: [(h, P(h from original ambiguous cross)), ...]
  Returns:
    p_reveal: overall probability that this tester will reveal which h we have
    unique_pheno: disjoint phenotype sets per h
    hit_prob: probability of hitting that disjoint set if true h
  """
  candidates = [g for g, _ in group]
  orig_probs = [p for _, p in group]
  total = sum(orig_probs)
  priors = {g: p/total for g, p in group}  # P(h) given this phenotype

  unique_pheno, hit_prob = disjoint_sets_for_tester(species, tester, candidates, transitions, genetics)

  # overall reveal probability = sum_h P(h) * r_h
  p_reveal = sum(priors[h]*hit_prob[h] for h in candidates)

  return p_reveal, unique_pheno, hit_prob

@dataclass
class TestSpec:
  phenotype: str                       # ambiguous phenotype we’re resolving
  tester: str                          # chosen tester genotype from state
  candidates: Tuple[str, ...]          # ambiguous genotypes
  reveal_probs: Dict[str, float]       # r_h: hit_prob per genotype
  p_reveal_overall: float              # overall P(reveal)

def choose_optimal_tester_for_group(species: str, state: FrozenSet[str], phenotype: str, group: List[Tuple[str, float]], transitions: "FlowerTransitions", genetics: "FlowerGenetics") -> Optional[TestSpec]:
  """
  For a single ambiguous phenotype group from AxB, pick the tester in `state`
  that maximizes overall reveal probability.
  """
  if not state:
    return None

  best_tester: Optional[str] = None
  best_p_reveal = 0.0
  best_hit: Dict[str, float] = {}

  for tester in state:
    p_reveal, unique_pheno, hit_prob = tester_reveal_probability(species, tester, group, transitions, genetics)
    if p_reveal > best_p_reveal:
      best_p_reveal = p_reveal
      best_tester = tester
      best_hit = hit_prob

  if best_tester is None or best_p_reveal == 0.0:
    return None

  candidates = tuple(g for g, _ in group)
  return TestSpec(phenotype = phenotype, tester = best_tester, candidates = candidates, reveal_probs = best_hit, p_reveal_overall = best_p_reveal)
