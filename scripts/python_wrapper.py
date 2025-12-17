from __future__ import annotations
from typing import Sequence, FrozenSet, List, Dict, Any
from mdp import State
from mcts import full_episode
from dag import build_genotype_dag, build_action_schedule

def _make_state(genotypes) -> State:
  """
  Convert an iterable of genotype strings into a State (frozenset).
  This is written to play nicely with reticulate:
  - If R passes a character vector, it becomes a Python list[str].
  """
  if genotypes is None: return frozenset()
  return frozenset(str(g) for g in genotypes)

def _make_targets(targets) -> List[FrozenSet[str]]:
  """
  Convert R-style targets into List[FrozenSet[str]].
  Expected forms from R:
  - list(c("geno1", "geno2"), c("geno3"))
  - or a list of length-1 character vectors if you’re using unique genotypes.
  """
  if targets is None: return []
  result: List[FrozenSet[str]] = []
  for group in targets: result.append(frozenset(str(g) for g in group))
  return result

def run_episode_for_shiny(species: str, targets, root_state, root_n_simulations: int = 1000, max_episode_steps: int = 1000, root_max_rollout_depth: int = 20, c: float = 2.0**0.5, min_n_simulations: int = 100, max_simulations_scale_factor: float = 0.0, min_depth_floor: int = 10, seed = None) -> Dict[str, Any]:
  """
  Wrapper around parallel_full_episode that returns a dict
  composed only of basic Python types (lists, dicts, numbers, strings)
  so R/reticulate can ingest it easily.
  Parameters are intentionally kept close to parallel_full_episode.
  """
  py_targets = _make_targets(targets)
  py_root_state = _make_state(root_state)

  episode = full_episode(species = species, targets = py_targets, root_state = py_root_state, root_n_simulations = root_n_simulations, max_episode_steps = max_episode_steps, root_max_rollout_depth = root_max_rollout_depth, c = c, min_n_simulations = min_n_simulations, max_simulations_scale_factor = max_simulations_scale_factor, min_depth_floor = min_depth_floor, seed = seed)

  trajectory = episode["trajectory"]
  final_state: State = episode["final_state"]

  # Build DAG + parallel schedule
  geno_dag = build_genotype_dag(trajectory, final_state)
  schedule = build_action_schedule(trajectory, geno_dag)

  # Convert trajectory into a step log that R can table-ize
  steps: List[Dict[str, Any]] = []
  states = [s for (s, a) in trajectory]
  actions = [a for (s, a) in trajectory]
  T = len(actions)
  for t in range(T):
    s_t = states[t]
    s_next = states[t + 1] if t < T - 1 else final_state
    new_genos = sorted(list(s_next.difference(s_t)))
    a_t = actions[t]
    wave = schedule.level_of_action.get(t, 0)

    steps.append({"step": int(t),
                  "wave": int(wave),
                  "parent1": a_t[0],
                  "parent2": a_t[1],
                  "available_before": sorted(list(s_t)),
                  "new_genotypes": new_genos,
                 })

  # Summarize waves
  waves: List[Dict[str, Any]] = []
  for wave_idx, step_ids in sorted(schedule.actions_by_level.items()):
    waves.append({"wave": int(wave_idx),
                  "steps": [int(s) for s in step_ids],
                 })

  # You can add more numeric stats here if desired:
  summary = {k: v for k, v in episode.items() if isinstance(v, (int, float, str, bool))}

  return {"steps": steps, "final_state": sorted(list(final_state)), "waves": waves, "summary": summary}
