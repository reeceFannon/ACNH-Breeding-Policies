from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict
import polars as pl

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
