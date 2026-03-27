from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Iterable
import networkx as nx
from qmdp import QuantumState, QuantumAction, QuantumFlower, QuantumFlowerMDP, create_flower_hash
from transitions import TransitionTensor

@dataclass
class QuantumFlowerDAG:
  graph: nx.DiGraph
  origin_step: Dict[str, int]                  # flower_hash -> step index (-1 for initial)
  produced_by: Dict[str, QuantumAction | None] # flower_hash -> action that created it
  flowers: Dict[str, QuantumFlower]            # flower_hash -> flower object

@dataclass
class QuantumActionSchedule:
  graph: nx.DiGraph
  level_of_action: Dict[int, int]
  actions_by_level: Dict[int, List[int]]

def build_hash_dag(trajectory: List[Tuple[QuantumState, QuantumAction, QuantumFlower]], final_state: QuantumState) -> QuantumFlowerDAG:
    if not trajectory: return QuantumFlowerDAG(graph = nx.DiGraph(), origin_step = {}, parents = {})

    states = [state for (state, action, child) in trajectory]
    actions = [action for (state, action, child) in trajectory]
    children = [child for (state, action, child) in trajectory]
    initial_state = states[0]

    origin_step: Dict[str, int] = {}
    parents: Dict[str, QuantumAction | None] = {}

    # Initial genotypes
    for flower in initial_state:
      origin_step[flower.hash] = -1
      parents[flower.hash] = None

    T = len(actions)
    for t in range(T): # Track new genotypes at each step
      action, child = actions[t], children[t]
      parent1, parent2 = action
      origin_step[child.hash] = t
      parents[child.hash] = action

    # Build hash graph
    G = nx.DiGraph()
    for hash in origin_step: G.add_node(hash) # Ensure all observed hashes are nodes

    for childHash, parents in parents.items():
      if parents is None: continue
      parent1, parent2 = parents
      for parent in (parent1, parent2): if parent not in G: G.add_node(parent.hash) # Ensure parents exist as nodes even if they never appeared in a state
      G.add_edge(parent1.hash, childHash)
      G.add_edge(parent2.hash, childHash)

    return QuantumFlowerDAG(graph = G, origin_step = origin_step, produced_by = parents)

def build_action_dag(trajectory: List[Tuple[QuantumState, QuantumAction, Offspring]], origin_step: Dict[str, int], initial_state: QuantumState | None = None) -> nx.DiGraph:
  n = len(trajectory)
  G = nx.DiGraph()

  if n == 0: return G
  if initial_state is None: initial_state = trajectory[0][0]

  G.add_nodes_from(range(n))

  # Add dependencies based on parent origins
  for j, (state, action, child) in enumerate(trajectory):
    parent1, parent2 = action
    for parent in (parent1, parent2):
      if parent in initial_state: continue # If parent genotype was available from the start, no dependency
      src = origin_step.get(parent.hash, None)
      if src is None: continue
      if src < j: G.add_edge(src, j) # Only dependencies from earlier actions make sense

  return G

def compute_action_schedule(G: nx.DiGraph) -> QuantumActionSchedule:
  if G.number_of_nodes() == 0: return QuantumActionSchedule(graph = G, level_of_action = {}, actions_by_level = {})

  level_of_action: Dict[int, int] = {}
  actions_by_level: Dict[int, List[int]] = {}

  # topological_generations yields layers of nodes with in-degree 0
  for lvl, nodes in enumerate(nx.topological_generations(G)):
    node_list = list(nodes)
    for i in node_list: level_of_action[i] = lvl
    actions_by_level[lvl] = sorted(node_list)

  return QuantumActionSchedule(graph = G, level_of_action = level_of_action, actions_by_level = actions_by_level)

def build_action_schedule(trajectory: List[Tuple[QuantumState, QuantumAction, Offspring]], geno_dag: QuantumFlowerDAG) -> QuantumActionSchedule:
  initial_state = trajectory[0][0] if trajectory else frozenset()
  action_graph = build_action_dag(trajectory = trajectory, origin_step = geno_dag.origin_step, initial_state = initial_state)
  return compute_action_schedule(action_graph)

def create_label(species: str, flower: QuantumFlower) -> str:
  return f"{species} | {flower.phenotype} | {flower.hash}"

def create_img_html(species: str, flower: QuantumFlower) -> str:
  return f"<img src='/imgs/{species}_{flower.phenotype}.png' class='picker-img'>"

Policy = Dict[]  

def build_policy_plan(species: str, trajectory: List[Tuple[QuantumState, QuantumAction, QuantumFlower]], transition_tensor: TransitionTensor) -> Policy:
  """
  Build an initial quantum policy object with waves and actions.

  Each action includes:
    - parent1 hash / phenotype / label / img_html
    - parent2 hash / phenotype / label / img_html
    - offspring rows by phenotype:
        prob, phenotype, hash, label, img_html
  """
  if not trajectory: return {"species": species, "waves": []}

  geno_dag = build_hash_dag(trajectory, final_state = trajectory[-1][0] | {trajectory[-1][2]})
  action_sched = build_action_schedule(trajectory, geno_dag)
  qmdp = QuantumFlowerMDP(species = species, transition_tensor = transition_tensor, targets = [])
  waves_out = []
  for wave_idx, action_idxs in sorted(action_sched.actions_by_level.items()):
    actions_out = []
    for action_idx in action_idxs:
      state, action, child = trajectory[action_idx]
      
      parent1, parent2 = action
      parent1_info = {"parent1_hash": parent1.hash,
                      "parent1_phenotype": parent1.phenotype,
                      "parent1_label": create_label(species, parent1),
                      "parent1_img_html": create_img_html(species, parent1)}
      parent2_info = {"parent2_hash": parent2.hash,
                      "parent2_phenotype": parent2.phenotype,
                      "parent2_label": create_label(species, parent2),
                      "parent2_img_html": create_img_html(species, parent2)}

      # Offspring phenotype distribution from this quantum action
      # Use the same distribution logic as qmdp
      offspring_dist = qmdp.breed_distribution(parent1, parent2)

      offspring_rows = []
      for phenotype, idxs in transition_tensor.phenotype_to_idx.items():
        prob = float(offspring_dist[idxs].sum().item())
        if prob <= 0: continue

        child_hash = create_flower_hash(phenotype, action)

        offspring_rows.append({"prob": prob,
                               "phenotype": phenotype,
                               "hash": child_hash,
                               "label": f"{species} | {phenotype} | {child_hash}",
                               "img_html": f"<img src='/imgs/{species}_{phenotype}.png' class='picker-img'>"})

      action_obj = {"action_idx": action_idx,
                    **parent1_info,
                    **parent2_info,
                    "offspring": offspring_rows}

      actions_out.append(action_obj)

    waves_out.append({"wave": wave_idx,
                      "actions": actions_out})

  return {"species": species, "waves": waves_out}

def add_keep_flags(policy: Policy, targets: List[str]) -> Policy:
  """
  Mark offspring rows as keep/discard based on whether they are used in future actions
  or match a target phenotype. Remove actions whose offspring are all discard.
  Repeat until stable.
  """
  targets = set(targets)

  changed = True
  while changed:
    changed = False

    # Flatten actions in wave order
    flat_actions = []
    for wave in policy["waves"]:
      for action in wave["actions"]: flat_actions.append((wave["wave"], action))

    # Precompute future parent hashes for each action position
    future_parent_hashes_by_idx = []
    for i in range(len(flat_actions)):
      future_hashes = set()
      for _, future_action in flat_actions[i + 1:]:
        future_hashes.add(future_action["parent1_hash"])
        future_hashes.add(future_action["parent2_hash"])
      future_parent_hashes_by_idx.append(future_hashes)

    # Mark keep flags
    for i, (_, action) in enumerate(flat_actions):
      future_hashes = future_parent_hashes_by_idx[i]
      for row in action["offspring"]: row["keep"] = (row["hash"] in future_hashes or row["phenotype"] in targets)

    # Remove actions with all-trash offspring
    for wave in policy["waves"]:
      new_actions = []
      for action in wave["actions"]:
        if any(row.get("keep", False) for row in action["offspring"]): new_actions.append(action)
        else: changed = True
      wave["actions"] = new_actions

    # Remove empty waves
    new_waves = []
    for wave in policy["waves"]:
      if wave["actions"]: new_waves.append(wave)
      else: changed = True
    policy["waves"] = new_waves

  # Renumber waves after pruning
  for new_idx, wave in enumerate(policy["waves"]): wave["wave"] = new_idx

  return policy
