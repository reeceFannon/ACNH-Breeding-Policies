from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable
import networkx as nx
from mdp import State, Action

@dataclass
class GenotypeDAG:
  """
  Path-specific genotype DAG for a single planner episode.
  Nodes: genotypes (strings).
  Edges: parent_genotype -> child_genotype (created by breeding).
  """
  graph: nx.DiGraph                      # genotype-level DAG
  origin_step: Dict[str, int]            # genotype -> step index (-1 for initial)
  parents: Dict[str, Action | None]      # genotype -> (parent1, parent2) or None


@dataclass
class ActionSchedule:
  """
  Action-level DAG and parallelization information.
  Nodes: action indices 0..T-1.
  Edges: i -> j means "action j depends on an output of action i".
  """
  graph: nx.DiGraph                      # action-level DAG
  level_of_action: Dict[int, int]        # action index -> parallel 'wave' index
  actions_by_level: Dict[int, List[int]] # wave index -> sorted list of actions at that wave


def build_genotype_dag(trajectory: List[Tuple[State, Action]], final_state: State) -> GenotypeDAG:
    """
    Reconstruct a genotype-level DAG from a single planner trajectory.
    Parameters
    ----------
    trajectory
        List of (state_t, action_t) pairs for t = 0..T-1, where each
        state_t is the set of genotypes BEFORE taking action_t.
    final_state
        The final state after the last action in the trajectory.
    Returns
    -------
    GenotypeDAG
        - origin_step[g] = -1 if g was in the initial state,
                           t  if g first appeared between state_t and state_{t+1}
                             (or between state_{T-1} and final_state for t = T-1).
        - parents[g] = (p1, p2) if g was created by crossing p1 x p2, else None.
        - graph: DiGraph with edges parent -> child for every created genotype.
    """
    if not trajectory:
      empty_graph = nx.DiGraph()
      return GenotypeDAG(graph=empty_graph, origin_step={}, parents={})

    states = [s for (s, a) in trajectory]
    actions = [a for (s, a) in trajectory]
    initial_state = states[0]

    origin_step: Dict[str, int] = {}
    parents: Dict[str, Action | None] = {}

    # Initial genotypes
    for g in initial_state:
      origin_step[g] = -1
      parents[g] = None

    T = len(actions)

    # Track new genotypes at each step
    for t in range(T):
      s_t = states[t]
      s_next = states[t + 1] if t < T - 1 else final_state

      new_genos = s_next.difference(s_t)
      for g in new_genos:
        # Only register the first time we ever see this genotype
        if g not in origin_step:
          origin_step[g] = t
          parents[g] = actions[t]

    # Build genotype graph
    G = nx.DiGraph()
    # Ensure all observed genotypes are nodes
    for g in origin_step: G.add_node(g)

    for child, pa in parents.items():
      if pa is None: continue
      p1, p2 = pa
      # Ensure parents exist as nodes even if they never appeared in a state
      for p in (p1, p2):
        if p not in G: G.add_node(p)
      G.add_edge(p1, child)
      G.add_edge(p2, child)

    return GenotypeDAG(graph=G, origin_step=origin_step, parents=parents)


def build_action_dag(trajectory: List[Tuple[State, Action]], origin_step: Dict[str, int], initial_state: State | None = None) -> nx.DiGraph:
  """
  Build the action-level DAG for a single trajectory.
  Parameters
  ----------
  trajectory
      List of (state_t, action_t) pairs (length T).
  origin_step
      Mapping genotype -> step index (-1 for genotypes available initially).
      Typically from GenotypeDAG.origin_step.
  initial_state
      The initial state. If None, uses trajectory[0][0].
  Returns
  -------
  nx.DiGraph
      Nodes are action indices 0..T-1.
      Edge i -> j means "action j uses a genotype created by action i".
  """
  n = len(trajectory)
  G = nx.DiGraph()

  if n == 0: return G
  if initial_state is None: initial_state = trajectory[0][0]

  G.add_nodes_from(range(n))

  # Add dependencies based on parent origins
  for j, (state, act) in enumerate(trajectory):
    p1, p2 = act
    for p in {p1, p2}:
      # If parent genotype was available from the start, no dependency
      if p in initial_state: continue
      src = origin_step.get(p, None)
      if src is None: continue
      if src < j: G.add_edge(src, j) # Only dependencies from earlier actions make sense

  return G


def compute_action_schedule(G: nx.DiGraph) -> ActionSchedule:
  """
  Given an action DAG, compute parallelizable 'waves' of actions using
  networkx.topological_generations.
  Parameters
  ----------
  G
      DiGraph over action indices.
  Returns
  -------
  ActionSchedule
      - level_of_action[i] = wave index (0 for earliest actions).
      - actions_by_level[w] = list of action indices in wave w.
  """
  if G.number_of_nodes() == 0:
    return ActionSchedule(graph=G, level_of_action={}, actions_by_level={})

  level_of_action: Dict[int, int] = {}
  actions_by_level: Dict[int, List[int]] = {}

  # topological_generations yields layers of nodes with in-degree 0
  for lvl, nodes in enumerate(nx.topological_generations(G)):
    node_list = list(nodes)
    for i in node_list:
      level_of_action[i] = lvl
    actions_by_level[lvl] = sorted(node_list)

  return ActionSchedule(graph = G, level_of_action = level_of_action, actions_by_level = actions_by_level)


def build_action_schedule(trajectory: List[Tuple[State, Action]], geno_dag: GenotypeDAG) -> ActionSchedule:
  """
  Convenience wrapper: build an action DAG from a trajectory and genotype DAG,
  then compute the parallelization schedule.
  Parameters
  ----------
  trajectory
      List of (state_t, action_t) pairs.
  geno_dag
      GenotypeDAG produced by build_genotype_dag.
  Returns
  -------
  ActionSchedule
  """
  initial_state = trajectory[0][0] if trajectory else frozenset()
  action_graph = build_action_dag(trajectory = trajectory, origin_step = geno_dag.origin_step, initial_state = initial_state)
  return compute_action_schedule(action_graph)
