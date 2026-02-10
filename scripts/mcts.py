import math
import random
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Sequence, FrozenSet
from transitions import FlowerTransitions, canonical_pair
from mdp import FlowerMDP, State, Action

def sample_next_state(mdp: FlowerMDP, state: State, action: Action, optimize: bool = False, transition_tensor: TransitionTensor = None) -> State:
  """Sample a single next state from the transition distribution."""
  if optimize:
    a, b = canonical_pair(*action)
    a, b = transition_tensor.genotype_to_idx[a], transition_tensor.genotype_to_idx[b]
    probs = transition_tensor.T[a, b, :]
    offspring = transition_tensor.idx_to_genotype[torch.multinomial(probs, 1)]
    return frozenset(set(state) | {offspring})
  
  dist: Dict[State, float] = mdp.next_state_distribution(state, action)
  states = list(dist.keys())
  probs = list(dist.values())
  return random.choices(states, weights = probs, k = 1)[0]

def random_rollout(mdp: FlowerMDP, start_state: State, max_depth: int = 20, optimize: bool = False, transition_tensor: TransitionTensor = None) -> float:
  """
  Random policy rollout from start_state.
  Returns reward = -steps to terminal (more negative if slower).
  """
  state = start_state
  for t in range(max_depth):
    if mdp.is_terminal(state):
      return float(-t)

    actions = mdp.available_actions(state)
    if not actions: # dead-end
      return float(-max_depth)

    action = random.choice(actions)
    state = sample_next_state(mdp, state, action, optimize = optimize, transition_tensor = transition_tensor)

  # didn't reach terminal within horizon
  return float(-max_depth)

@dataclass
class MCTSNode:
  state: State
  parent: Optional["MCTSNode"] = None
  action_from_parent: Optional[Action] = None

  children: Dict[Action, "MCTSNode"] = field(default_factory = dict)
  untried_actions: List[Action] = field(default_factory = list)

  visits: int = 0
  total_reward: float = 0.0
  total_sq_reward: float = 0.0

  def is_fully_expanded(self) -> bool:
    return len(self.untried_actions) == 0

  def best_child(self, c: float = math.sqrt(2.0)) -> "MCTSNode":
    """
    UCT selection: argmax over children of (Q / N + c * sqrt(log Np / Nc)).
    """
    best_score = -float("inf")
    best = None
    for child in self.children.values():
      if child.visits == 0:
        uct = float("inf")
      else:
        exploit = child.total_reward / child.visits
        explore = c*math.sqrt(math.log(self.visits)/child.visits)
        uct = exploit + explore
      if uct > best_score:
        best_score = uct
        best = child
    return best

def mcts_search(mdp: FlowerMDP, root_state: State, n_simulations: int = 1000, max_rollout_depth: int = 20, c: float = math.sqrt(2.0), optimize: bool = False, transition_tensor: TransitionTensor = None) -> MCTSNode:
  """
  Run MCTS from the root_state and return the root node
  with its tree of children filled in.
  """
  root = MCTSNode(state = root_state, parent = None, action_from_parent = None, untried_actions = mdp.available_actions(root_state))

  for _ in range(n_simulations):
    node = root

    # 1. SELECTION: descend tree using UCT
    while node.is_fully_expanded() and node.children:
      node = node.best_child(c)

    # 2. EXPANSION: expand one untried action (if any and not terminal)
    if not mdp.is_terminal(node.state) and node.untried_actions:
      action = node.untried_actions.pop()
      next_state = sample_next_state(mdp, node.state, action, optimize = optimize, transition_tensor = transition_tensor)

      child = MCTSNode(state = next_state, parent = node, action_from_parent = action, untried_actions = mdp.available_actions(next_state))
      node.children[action] = child
      node = child  # next we rollout from child

    # 3. SIMULATION (ROLLOUT): from this leaf node
    reward = random_rollout(mdp, node.state, max_depth = max_rollout_depth, optimize = optimize, transition_tensor = transition_tensor)

    # 4. BACKPROPAGATION
    while node is not None:
      node.visits += 1
      node.total_reward += reward
      node.total_sq_reward += reward**2
      node = node.parent

  return root

def full_episode(species: str, targets: Sequence[FrozenSet[str]], root_state: State, root_n_simulations: int = 1000, max_episode_steps: int = 1000, root_max_rollout_depth: int = 20, c: float = math.sqrt(2.0), min_n_simulations: int = 100, max_simulations_scale_factor: float = 0.0, min_depth_floor: int = 10, seed: int | None = None, optimize: bool = False) -> Dict[str, object]:
  state = root_state
  total_steps = 0
  trajectory: List[Tuple[State, Action]] = []
  current_max_rollout_depth = root_max_rollout_depth
  current_n_simulations = root_n_simulations
  step_stats = StepStats()
  transitions = FlowerTransitions()
  success = False
  episode_seed = seed if seed is not None else random.randint(1, 2**31 - 1)
  random.seed(episode_seed)
  
  if optimize:
    from transitions import TransitionTensor, TransitionTensorBuilder
    transition_tensor = TransitionTensorBuilder().build_transition_tensor(mdp.species)
  else:
    transition_tensor = None
    
  while (not success) and (total_steps < max_episode_steps):
    if current_max_rollout_depth <= 0:
      break

    # --- parallel root MCTS from current state ---
    mdp = FlowerMDP(species = species, transitions = transitions, targets = targets)
    root = mcts_search(mdp, state, current_n_simulations, current_max_rollout_depth, c, optimize, transition_tensor)
    root_stats = extract_root_action_stats(root)
    best_action = best_root_action_from_stats(root_stats)

    if best_action is None:
      break  # no legal actions

    trajectory.append((state, best_action))

    # sample next state in the *main* process
    next_state = sample_next_state(mdp, state, best_action, optimize = optimize, transition_tensor = transition_tensor)
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


@dataclass
class StepStats:
  n: int = 0
  sum_steps: float = 0.0
  sum_sq_steps: float = 0.0

  def update(self, n: int, sum_steps: float, sum_sq_steps: float) -> None:
    self.n = n
    self.sum_steps = sum_steps
    self.sum_sq_steps = sum_sq_steps

  @property
  def mean(self) -> Optional[float]:
    if self.n == 0:
      return None
    return self.sum_steps/self.n

  @property
  def variance(self) -> Optional[float]:
    if self.n == 0:
      return None
    m = self.mean
    return (self.sum_sq_steps/self.n) - m**2 # E(X^2) - (E(X))^2

def extract_root_action_stats(root: MCTSNode) -> Dict[Action, Dict[str, float]]:
  """
  Summarize stats for each root action:
  returns {action: {"visits": v, "total_reward": r}}.
  """
  stats: Dict[Action, Dict[str, float]] = {}
  for action, child in root.children.items():
    stats[action] = {"visits": child.visits,
                      "total_reward": child.total_reward,
                      "total_sq_reward": child.total_sq_reward}
  return stats

def best_root_action_from_stats(stats: Dict[Action, Dict[str, float]]) -> Optional[Action]:
  if not stats:
    return None
  return max(stats.items(), key=lambda kv: kv[1]["visits"])[0]
