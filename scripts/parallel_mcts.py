import math
import random
import multiprocessing as mp
from typing import Iterable, Dict, List, Tupple, Sequence, FrozenSet
from transitions import FlowerTransitions
from mdp import FlowerMDP, State, Action
from mcts import (mcts_search, extract_root_action_stats, best_root_action_from_stats)

def _worker_mcts_run(species: str, targets: Sequence[FrozenSet[str]], root_state: State, n_simulations: int, max_rollout_depth: int, c: float = math.sqrt(2.0), seed: int | None = None) -> Dict[Action, Dict[str, float]]:
  """
  Worker function: builds its own MDP instance, runs MCTS,
  and returns root action stats.
  """
  if seed is not None:
    random.seed(seed)

  transitions = FlowerTransitions()
  mdp = FlowerMDP(species = species, transitions = transitions, targets = targets)
  root = mcts_search(mdp, root_state = root_state, n_simulations = n_simulations, max_rollout_depth = max_rollout_depth, c = c)
  return extract_root_action_stats(root)

def parallel_mcts_root(species: str, targets: Sequence[FrozenSet[str]], root_state: State, total_simulations: int = 1000, max_rollout_depth: int = 20, n_workers: int = 4, c: float = math.sqrt(2.0), seeds: List[int] | None = None) -> tuple[Action | None, Dict[Action, Dict[str, float]]]:
  """
  Root-parallel MCTS:
    - splits total_simulations across n_workers,
    - runs MCTS in each worker,
    - aggregates root action stats,
    - returns best action and aggregated stats.
  """
  sims_per_worker = total_simulations // n_workers
  if sims_per_worker == 0:
    sims_per_worker = 1

  if seeds is None: seeds = list(range(n_workers))
  elif len(seeds) < n_workers: raise Exception(f"Not enough seeds. Seeds requires length {n_workers}. You have {len(seeds}")
  else: seeds = seeds[:n_workers]

  with mp.Pool(processes=n_workers) as pool:
    args = [(species, targets, root_state, sims_per_worker, max_rollout_depth, c, None if i is None else i) for i in seeds]
    results = pool.starmap(_worker_mcts_run, args)

  # Aggregate stats
  aggregated: Dict[Action, Dict[str, float]] = {}
  for stats in results:
    for action, s in stats.items():
      if action not in aggregated:
        aggregated[action] = {"visits": 0.0, "total_reward": 0.0}
      aggregated[action]["visits"] += s["visits"]
      aggregated[action]["total_reward"] += s["total_reward"]

  best_action = best_root_action_from_stats(aggregated)
  return best_action, aggregated
