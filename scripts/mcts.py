import math
import random
from typing import Dict
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from mdp import FlowerMDP, State, Action

def sample_next_state(mdp: FlowerMDP, state: State, action: Action) -> State:
  """Sample a single next state from the transition distribution."""
  dist: Dict[State, float] = mdp.next_state_distribution(state, action)
  states = list(dist.keys())
  probs = list(dist.values())
  return random.choices(states, weights = probs, k = 1)[0]

def random_rollout(mdp: FlowerMDP, start_state: State, max_depth: int = 20) -> float:
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
    state = sample_next_state(mdp, state, action)

  # didn't reach terminal within horizon
  return float(-max_depth)

from dataclasses import dataclass, field
from typing import Optional, Dict, List
import math

@dataclass
class MCTSNode:
  state: State
  parent: Optional["MCTSNode"] = None
  action_from_parent: Optional[Action] = None

  children: Dict[Action, "MCTSNode"] = field(default_factory = dict)
  untried_actions: List[Action] = field(default_factory = list)

  visits: int = 0
  total_reward: float = 0.0

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

def mcts_search(mdp: FlowerMDP, root_state: State, n_simulations: int = 1000, max_rollout_depth: int = 20, c: float = math.sqrt(2.0)) -> MCTSNode:
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
      next_state = sample_next_state(mdp, node.state, action)

      child = MCTSNode(state = next_state, parent = node, action_from_parent = action, untried_actions = mdp.available_actions(next_state))
      node.children[action] = child
      node = child  # next we rollout from child

    # 3. SIMULATION (ROLLOUT): from this leaf node
    reward = random_rollout(mdp, node.state, max_depth = max_rollout_depth)

    # 4. BACKPROPAGATION
    while node is not None:
      node.visits += 1
      node.total_reward += reward
      node = node.parent

  return root

def extract_root_action_stats(root: MCTSNode) -> Dict[Action, Dict[str, float]]:
  """
  Summarize stats for each root action:
  returns {action: {"visits": v, "total_reward": r}}.
  """
  stats: Dict[Action, Dict[str, float]] = {}
  for action, child in root.children.items():
    stats[action] = {"visits": child.visits,
                      "total_reward": child.total_reward}
  return stats

def best_root_action_from_stats(stats: Dict[Action, Dict[str, float]]) -> Optional[Action]:
  if not stats:
    return None
  return max(stats.items(), key=lambda kv: kv[1]["visits"])[0]
