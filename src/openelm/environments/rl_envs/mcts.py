import numpy as np
import math
import random
from typing import Optional, Callable
from collections import defaultdict


class MCTSNode:
    def __init__(self,
                 value_fn,
                 parent_action, 
                 parent, 
                 terminal=False,
                 exp_coef=2**(1/2),
                 reward=0,):
        self.value_fn = value_fn
        self.parent_action = parent_action
        self.parent = parent
        self.children = []
        self.n: int = 0  # Num visits to node
        self.results: List[float] = [] # Return of each visit
        self.taken_actions = set()
        self.terminal = terminal  # True if game is in terminal state
        self.exp_coef = exp_coef  # Exploration coefficient
        self.reward = reward

    def expand(self, rl_env):
        """
        Expands a new node to add into the MCTS tree.
        """
        assert not self.terminal
        actions = set(rl_env.get_actions())
        new_actions = list(actions - self.taken_actions)
        action = np.random.choice(new_actions)
        self.taken_actions.add(action)
        observation, reward, done, _, _ = rl_env.step(action)
        new_node = MCTSNode(value_fn=self.value_fn,
                            parent_action=action,
                            parent=self,
                            terminal=done,
                            exp_coef=self.exp_coef,
                            reward=reward,)
        self.children.append(new_node)
        return new_node

    def is_unexpanded(self):
        return len(self.children) == 0

    def is_fully_expanded(self, rl_env):
        if len(self.taken_actions - set(rl_env.get_actions())) > 0:
            raise ValueError(f"Too many actions!!!\n\n\
Node: {self.taken_actions}\n\n\
rl_env: {rl_env.get_actions()}\n\n\
diff: {self.taken_actions - set(rl_env.get_actions())}\n\n\
state:\n\n {rl_env.render()}")
        return len(self.taken_actions) == len(rl_env.get_actions())

    def _selection_policy(self, rl_env):
        """
        Implements UCT exploration policy during selection phase.
        """
        weights = [np.mean(node.results) + self.exp_coef*np.sqrt(2*np.log(self.n)/node.n) for node in self.children]
        weights = np.nan_to_num(weights, nan=0) + 1e-9*np.random.randn(len(weights))  # Possibly nan if executing in parallel
        next_node = self.children[np.argmax(weights)]
        return next_node

    def select(self, rl_env):
        """
        Selects a node to rollout from.
        """
        current_node = self
        while not current_node.is_unexpanded():
            if not current_node.is_fully_expanded(rl_env):
                return current_node.expand(rl_env)
            else:
                current_node = current_node._selection_policy(rl_env)
                rl_env.step(current_node.parent_action)
        if not current_node.terminal:
            return current_node.expand(rl_env)
        else:
            return current_node

    def _rollout_policy(self, rl_env):
        """
        Policy used to select moves during rollout. Currently random.
        """
        return np.random.choice(list(rl_env.get_actions()))

    def rollout(self, rl_env, depth: int) -> float:
        """
        Rollout the current rl_env.
        + rl_env: rl_env to rollout
        + depth: depth to rollout to
        Returns: return of rollout.
        Note: This assumes the current state of rl_env is the called node
        """
        # TODO(dahoas): maybe track move history to ensure rl_env agrees with node
        # Don't want to store a state in the node for performance/memory reasons
        if self.terminal:
            return self.reward
        for i in range(depth):
            action = self._rollout_policy(rl_env)
            observation, reward, done, _, _ = rl_env.step(action)
            if done:
                return reward if reward != 0 else self.value_fn(observation)
        return self.value_fn(observation)

    def backprop(self, result):
        self.n += 1
        self.results.append(result)
        if self.parent:
            self.parent.backprop(result)
