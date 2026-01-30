import math
import random
import multiprocessing as mp
from typing import Iterable, Dict, List, Tuple, Sequence, FrozenSet
from transitions import FlowerTransitions
from mdp import FlowerMDP, State, Action
from mcts import (sample_next_state, mcts_search, extract_root_action_stats, best_root_action_from_stats, StepStats)

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
  elif len(seeds) < n_workers: raise Exception(f"Not enough seeds. Seeds requires length {n_workers}. You have {len(seeds)}")
  else: seeds = seeds[:n_workers]

  args = [(species, targets, root_state, sims_per_worker, max_rollout_depth, c, None if i is None else i) for i in seeds]
  ctx = mp.get_context("spawn")
  with ctx.Pool(processes=n_workers) as pool:
    results = pool.starmap(_worker_mcts_run, args)

  # Aggregate stats
  aggregated: Dict[Action, Dict[str, float]] = {}
  for stats in results:
    for action, s in stats.items():
      if action not in aggregated:
        aggregated[action] = {"visits": 0.0, "total_reward": 0.0, "total_sq_reward": 0.0}
      aggregated[action]["visits"] += s["visits"]
      aggregated[action]["total_reward"] += s["total_reward"]
      aggregated[action]["total_sq_reward"] += s["total_sq_reward"]

  best_action = best_root_action_from_stats(aggregated)
  return best_action, aggregated

def parallel_full_episode(species: str, targets: Sequence[FrozenSet[str]], root_state: State, root_n_simulations: int = 1000, max_episode_steps: int = 1000, root_max_rollout_depth: int = 20, n_workers: int = 4, c: float = math.sqrt(2.0), min_n_simulations: int = 100, max_simulations_scale_factor: float = 0.0, min_depth_floor: int = 10, seeds: List[int] | None = None) -> Dict[str, object]:
  """
  High-level episode driver that uses *parallel* MCTS at each step.
  Loop:
    - while not terminal and steps < max:
        * pick action via parallel_mcts_root
        * sample next state
  Then update (state, action) -> StepStats with steps-to-finish.

  Returns a dict with:
    - 'trajectory'  : [(state, action), ...]
    - 'final_state' : last state
    - 'total_steps' : number of actions taken
    - 'success'     : mdp.is_terminal(final_state)
    - 'stats'       : the (possibly updated) action_step_stats
  """
  state = root_state
  total_steps = 0
  trajectory: List[Tuple[State, Action]] = []
  current_max_rollout_depth = root_max_rollout_depth
  current_n_simulations = root_n_simulations
  step_stats = StepStats()
  transitions = FlowerTransitions()
  mdp = FlowerMDP(species = species, transitions = transitions, targets = targets)
  success = False
  while (not success) and (total_steps < max_episode_steps):
    if current_max_rollout_depth <= 0:
      break

    # --- parallel root MCTS from current state ---
    step_seeds = seeds if seeds is not None else [random.randint(1, 2**31 - 1) for i in range(n_workers)]
    best_action, root_stats = parallel_mcts_root(species = species, targets = targets, root_state = state, total_simulations = current_n_simulations, max_rollout_depth = current_max_rollout_depth, n_workers = n_workers, c = c, seeds = step_seeds)

    if best_action is None:
      break  # no legal actions

    trajectory.append((state, best_action))

    # sample next state in the *main* process
    next_state = sample_next_state(mdp, state, best_action)
    state = next_state
    total_steps += 1
    print(f"Finished step {total_steps}: Chose action {best_action} with {current_n_simulations} rollouts at a depth of {current_max_rollout_depth}")

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
    
    success = mdp.is_terminal(state)

  return {"trajectory": trajectory, "final_state": state, "total_steps": total_steps, "success": success}
