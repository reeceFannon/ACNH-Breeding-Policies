from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, FrozenSet, Optional
import polars as pl
import torch

def canonical_pair(a: str, b: str) -> tuple[str, str]:
  """Make sure (A,B) and (B,A) map to the same key."""
  return tuple(sorted((a, b), reverse = True))

@dataclass
class FlowerDataPaths:
  transitions_csv: str = "data/breeding_transitions.csv"
  genetics_csv: str = "data/ACNH_flower_genetics.csv"

class FlowerTransitions:
  """
  Lazy interface to breeding_transitions.csv using Polars.
  Provides offspring_distribution(species, parent1, parent2)
  and caches per-species data in memory.
  """
  def __init__(self, paths: FlowerDataPaths | None = None):
    self.paths = paths or FlowerDataPaths()

  @lru_cache(maxsize = None)
  def _species_df(self, species: str) -> pl.DataFrame:
    """
    Load only rows for a given species from the big CSV.
    Cached in memory after first use.
    """
    species = species.lower()
    lf = (pl.scan_csv(self.paths.transitions_csv).filter(pl.col("species") == species))
    return lf.collect()

  def offspring_distribution(self, species: str, parent1: str, parent2: str) -> Dict[str, float]:
    """
    Return a dict: {offspring_genotype: prob} for this species + parent pair.
    If the pair was never precomputed (e.g., invalid parents),
    returns an empty dict.
    """
    species = species.lower()
    p1, p2 = canonical_pair(parent1, parent2)

    df_sp = self._species_df(species)
    # Modified filter to check for both (p1, p2) and (p2, p1) in the dataframe
    df_pair = df_sp.filter((pl.col("parent1") == p1) & (pl.col("parent2") == p2))

    if df_pair.height == 0:
      return {}

    # df_pair: columns ['species','parent1','parent2','offspring','prob']
    return dict(zip(df_pair["offspring"].to_list(), df_pair["prob"].to_list()))

@dataclass
class FlowerGenetics:
  paths: FlowerDataPaths

  def __post_init__(self):
    self._df = pl.read_csv(self.paths.genetics_csv)

  def genotypes_for_phenotypes(self, species: str, phenotypes: tuple[str, ...]) -> List[FrozenSet[str]]:
    """
    Given a species and a tuple of phenotype names,
    return a list of frozensets of genotypes:
        [ G(phenotype_1), G(phenotype_2), ... ]
    where each G(...) is the set of genotypes for that phenotype.
    Order matches the order of `phenotypes`.
    """
    species = species.lower()
    norm_phens = tuple(p.strip().lower() for p in phenotypes)

    result: List[FrozenSet[str]] = []
    for ph in norm_phens:
      subset = self._df.filter((pl.col("flower") == species) & (pl.col("phenotype") == ph))
      genos = frozenset(subset["genotype"].to_list())
      result.append(genos)

    return result

  def phenotypes_for_genotypes(self, species: str, genotypes: List[str]) -> List[str]:
      """
      Given a species and a list of genotypes,
      return a list of frozensets of phenotypes:
          [ Q(phenotype_1), Q(phenotype_2), ... ]
      where each Q(...) is the phenotype for that genotype.
      Order matches the order of `genotypes`.
      """
      species = species.lower()
  
      result: List[FrozenSet[str]] = []
      for g in genotypes:
        subset = self._df.filter((pl.col("flower") == species) & (pl.col("genotype") == g))
        phenos = tuple(subset["phenotype"].to_list())
        result.append(phenos)
  
      return result

@dataclass(frozen = True)
class TransitionTensor:
    T: torch.Tensor                      # [N, N, N] = [parent1, parent2, offspring]
    genotype_to_idx: Dict[str, int]      # genotype -> index (0..N-1)
    idx_to_genotype: List[str]           # index -> genotype

class TransitionBuilder:
  def __init__(self, paths: FlowerDataPaths | None = None):
    self.paths = paths or FlowerDataPaths()

    def build_transition_tensor(self, species: str, *, device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32) -> TransitionTensor:
      """
      Builds T[parent1, parent2, offspring] = P(offspring | parent1,parent2)
      from self.paths.transitions_csv using polars.
      """
      lf = pl.scan_csv(self.paths.transitions_csv)
      lf = lf.filter(pl.col("species") == species).select(["parent1", "parent2", "offspring", "prob"])

      df = lf.collect()
      if df.height == 0:
        raise ValueError(f"No rows found for species={species!r} in {self.paths.transitions_csv}")

      genotypes = sorted(set(df["parent1"].to_list()) | set(df["parent2"].to_list()) | set(df["offspring"].to_list()))
      N = len(genotypes)
      genotype_to_idx = {g: i for i, g in enumerate(genotypes)}
      idx_to_genotype = genotypes

      # Map genotype strings to integer indices using replace_strict
      # (replace_strict will error if a value isn't in the mapping)
      mapping_expr = pl.col  # small alias
      df_idx = df.with_columns(mapping_expr("parent1").replace_strict(genotype_to_idx).cast(pl.UInt8).alias("p1"),
                               mapping_expr("parent2").replace_strict(genotype_to_idx).cast(pl.UInt8).alias("p2"),
                               mapping_expr("offspring").replace_strict(genotype_to_idx).cast(pl.UInt8).alias("off"),
                               mapping_expr("prob").cast(pl.Float32).alias("prob_f32")).select(["p1", "p2", "off", "prob_f32"])
    
      p1 = torch.tensor(df_idx["p1"].to_numpy(), device = device, dtype = torch.long)
      p2 = torch.tensor(df_idx["p2"].to_numpy(), device = device, dtype = torch.long)
      off = torch.tensor(df_idx["off"].to_numpy(), device = device, dtype = torch.long)
      prob = torch.tensor(df_idx["prob_f32"].to_numpy(), device = device, dtype = dtype)

      # Allocate and fill T
      T = torch.zeros((N, N, N), device = device, dtype = dtype)
      T[p1, p2, off] = prob
      T[p2, p1, off] = prob

      pair_sums = T.sum(dim=2)  # [N, N]
      present = pair_sums > 0
      if present.any():
        max_err = (pair_sums[present] - 1.0).abs().max().item()
        if max_err > 1e-4:
          raise ValueError(
              f"Offspring probability sums not ~1 for some (parent1,parent2) pairs. "
              f"max_err={max_err:.6g}"
          )

      if not torch.allclose(T, T.transpose(0, 1), atol = 1e-6, rtol = 0):
        raise ValueError("T is not symmetric in parent1/parent2 after fill.")

      return TransitionTensor(T=T, genotype_to_idx=genotype_to_idx, idx_to_genotype=idx_to_genotype)
