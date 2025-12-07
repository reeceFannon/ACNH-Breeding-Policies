import math
import random
import multiprocessing as mp
from typing import Iterable, Dict, List, Tuple, Sequence, FrozenSet
from transitions import FlowerTransitions
from mdp import FlowerMDP, State, Action
from mcts import (sample_next_state, mcts_search, extract_root_action_stats, best_root_action_from_stats, StepStats, ActionStepStats)

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
        aggregated[action] = {"visits": 0.0, "total_reward": 0.0}
      aggregated[action]["visits"] += s["visits"]
      aggregated[action]["total_reward"] += s["total_reward"]

  best_action = best_root_action_from_stats(aggregated)
  return best_action, aggregated

def parallel_full_episode(species: str, targets: Sequence[FrozenSet[str]], root_state: State, total_simulations: int = 1000, max_episode_steps: int = 1000, root_max_rollout_depth: int = 20, n_workers: int = 4, c: float = math.sqrt(2.0), action_step_stats: ActionStepStats | None = None, min_depth_floor: int = 10, seeds: List[int] | None = None) -> Dict[str, object]:
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
  if action_step_stats is None:
    action_step_stats = {}

  transitions = FlowerTransitions()
  mdp = FlowerMDP(species = species, transitions = transitions, targets = targets)

  state = root_state
  total_steps = 0
  trajectory: List[Tuple[State, Action]] = []

  while (not mdp.is_terminal(state)) and (total_steps < max_episode_steps):
    remaining = max_episode_steps - total_steps

    # --- dynamic rollout depth from (state, action) step stats ---
    rollout_depth = min(remaining, root_max_rollout_depth)
    actions = mdp.available_actions(state)

    if actions:
      candidate_depths: List[float] = []
      for a in actions:
        key = (state, a)
        st = action_step_stats.get(key)
        if st is not None and st.mean is not None and st.variance is not None:
          mu = st.mean
          sigma = math.sqrt(max(st.variance, 0.0))
          candidate_depths.append(mu + 4*sigma) # from Chebyshev's Inquality

      if candidate_depths:
        est = max(min(candidate_depths), min_depth_floor)
        rollout_depth = int(min(remaining, math.ceil(est)))

    if rollout_depth <= 0:
      break

    # --- parallel root MCTS from current state ---
    step_seeds = seeds if seeds is not None else [random.randint(1, 1000000) for i in range(n_workers)]
    best_action, root_stats = parallel_mcts_root(species = species, targets = targets, root_state = state, total_simulations = total_simulations, max_rollout_depth = rollout_depth, n_workers = n_workers, c = c, seeds = step_seeds)

    if best_action is None:
      break  # no legal actions

    trajectory.append((state, best_action))

    # sample next state in the *main* process
    next_state = sample_next_state(mdp, state, best_action)
    state = next_state
    total_steps += 1
    print(f"Finished step {total_steps}: Chose action {best_action} with rollout depth {rollout_depth}")

  # --- update (state, action) -> steps-to-finish stats for this episode ---
  for t, (s_t, a_t) in enumerate(trajectory):
    steps_from_here = total_steps - t
    key = (s_t, a_t)
    stats = action_step_stats.get(key)
    if stats is None:
      stats = StepStats()
      action_step_stats[key] = stats
    stats.update(steps_from_here)

  success = mdp.is_terminal(state)

  return {"trajectory": trajectory, "final_state": state, "total_steps": total_steps, "success": success, "stats": action_step_stats}
