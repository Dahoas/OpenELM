import json
from abc import abstractmethod
import re
import warnings
from typing import Optional, List, Any
from functools import reduce
from copy import deepcopy
from enum import Enum
from dataclasses import asdict, dataclass
from time import time
import multiprocessing as mp

import numpy as np

from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.environments.rl_envs.mcts import MCTSNode
from openelm.algorithms.fun_search import Program
from openelm.environments.rl_env_util.env_wrappers import get_wrapped_env

import gymnasium as gym


class TaskType(Enum):
    POLICY = 1
    VALUE = 2

    @classmethod
    def from_string(cls, name):
        name_to_val = {val.name: val for val in cls}
        if name_to_val.get(name.upper(), None):
            return name_to_val[name.upper()]
        else:
            raise ValueError(f"Unknown name: {name}!!!")


def get_task_prompt(task_type: TaskType):
    if  task_type == TaskType.POLICY:
        return """\
You are responsible for designing a decision policy to solve the following task: 
{task_description}\n\n\
You will write a python `Policy()`, which should be initializable without any parameters from the user, object which has two methods:
- `def act(observation)` which takes in an observation and returns an action.
- `update(observation, action, reward, next_observation)` which takes in the current observation, \
chosen action, reward, and next_observation and updates any persistent memory/state between observations.
- `notes: list[str]` which is a list of signals tracked by the policy during execution. \
The signals collected should be designed by you to improve future iterations of this policy.
- `produce_report() -> str` which uses the collected notes to produce a few line summary you will receive on the \
policy's performance. You should try to use this report to collect statistics understanding how your policy fails \
or can be improved. In particular you should ensure you understand the dynamics of the environment, i.e. if the actions \
you take actually result in the state you expect.
Note: You should not assume any exploration outside of what is learned during the agent's single rollout in \
the environment. This means you should not rely on Q-learning, etc.\n\n\
The observation space is defined formally as: 
{observation_description}\n\n\
The action space is defined formally as:
{action_description}\n\n\
The rewards are defined formally as:
{reward_description}\n\n\
Consider the following example action sequence to familiairize yourself with the env dynamics\n\n\
{action_exemplar}\n\n\
You are allowed to use any python library you want but should not assume access \
to any other external resources (such as models with downloadable weights) unless otherwise specified. \
In particular you can assume access to the following APIs: \
{api_description}\n\n\
You should only write the Policy class and nothing else. \
You are encouraged to be as creative as possible, do not simply copy one of the exemplars if given. \
Your policy should also be robust to adversity. If it finds itself getting stuck, repeating the same moves, \
it should try something new. Exploration is encouraged, but keep in mind the goal and the action preconditions.\
Make sure when you are engaging in exploration, it is diverse and incentivizing discovery of new states.\n\
Thank carefully about the order of steps you propose. Executing the right steps in the wrong order will be costly\n\
All code should be written in a single, large code block.
"""
    elif task_type == TaskType.VALUE:
        return """\
You are responsible for designing a value function to solve the following task: 
{task_description}\n\n\
You will write a python `Value`, which should be initializable without any parameters from the user, object which has one method:
- `def value(observation)` which takes in an observation and returns the value of the observation. The \
output should be normalized between -1 and 1 \.
Note: You should not assume any exploration outside of what is learned during the agent's single rollout in \
the environment. This means you should not rely on Q-learning, requiring extra exploration.\n\n\
The observation space is defined formally as: 
{observation_description}\n\n\
You are allowed to use any python library you want but should not assume access \
to any other external resources (such as models with downloadable weights) unless otherwise specified. \
In particular you can assume access to the following APIs: \
{api_description}\n\n\
You should only write the Value class and nothing else. \
Improve the given exemplar as much as possible, filling in as many details as you can. \
All code should be written in a single, large code block.
"""


class RLPolicy:
    def __init__(self, 
                 rl_env,
                 policy,):
        self.rl_env = rl_env
        self.policy = policy

    def make_env_copy(self, **kwargs):
        return self.rl_env.deepcopy(**kwargs) if hasattr(self.rl_env, "deepcopy") else deepcopy(self.rl_env)

    def restore_env(self, rl_env_copy):
        if hasattr(rl_env_copy, "restore"):
            rl_env_copy.restore()

    def act(self, observation):
        return self.policy.act(observation)
    
    def update(self, old_observation, action, reward, observation):
        return self.policy.update(old_observation, action, reward, observation)


class RLValuePolicy(RLPolicy):
    """
    Implements methods of contructing policies from value function
    """
    def __init__(self,
                 rl_env,
                 value_fn,
                 method: str="mcts", 
                 depth: int=10, 
                 time_limit: float=30,
                 rollout_limit: int=10000,
                 num_procs=1,):
        self.rl_env = rl_env
        self.value_fn = value_fn.value
        self.method = method
        self.depth = depth
        self.time_limit = time_limit
        self.rollout_limit = rollout_limit
        self.num_procs = num_procs
        if self.method == "mcts":
            self.mcts_root: MCTSNode = MCTSNode(value_fn=self.value_fn,
                                                parent_action=None,
                                                parent=None,)

    def _value_only_policy(self, observation):
        action_dict = {action: 0 for action in self.rl_env.get_actions()}
        for action in action_dict:
            rl_env_copy = self.make_env_copy()  # Make (fake) copy of current state
            observation, reward, terminated, _, _ = rl_env_copy.step(action)
            value = self.value_fn(observation)
            action_dict[action] = reward if reward != 0 else value
            self.restore_env(rl_env_copy)  # Need to restore original state as this is not a true copy
        action_ind = np.argmax(list(action_dict.values()))
        return list(action_dict.keys())[action_ind]

    def _mcts_policy(self, observation):
        def do_rollouts(nodes, rl_env, depth, results, batch_size, rank):
            for i, node in enumerate(nodes):
                rl_env_copy = self.make_env_copy(mode="single")
                ret = node.rollout(rl_env_copy, depth=depth)
                results[rank * batch_size + i] = ret
                self.restore_env(rl_env_copy)

        t = time()
        if self.num_procs == 1:
            # Rollout MCTS tree
            for i in range(self.rollout_limit):
                elapsed = time() - t
                if elapsed > self.time_limit: break
                rl_env_copy = self.make_env_copy(mode="single")
                node = self.mcts_root.select(rl_env_copy)            
                ret = node.rollout(rl_env_copy, depth=self.depth)
                node.backprop(ret)
                self.restore_env(rl_env_copy)
        else:
            # TODO(dahoas): This impl is incorrect. Faster but degrades perf
            batch_size = self.rollout_limit // self.num_procs
            results = mp.Array("d", [np.nan for _ in range(self.rollout_limit)])
            nodes, procs = [], []
            for i in range(self.rollout_limit):
                rl_env_copy = self.make_env_copy(mode="single")
                node = self.mcts_root.select(rl_env_copy)
                nodes.append(node)
                self.restore_env(rl_env_copy)
            for i in range(self.num_procs):
                proc_nodes = nodes[i*batch_size:(i+1)*batch_size]
                p = mp.Process(target=do_rollouts, args=(proc_nodes, self.rl_env, self.depth, results, batch_size, i))
                procs.append(p)
                p.start()
            for i, p in enumerate(procs):
                p.join()
                rets = results[i*batch_size:(i+1)*batch_size]
                ret_nodes = nodes[i*batch_size:(i+1)*batch_size]
                for ret, node in zip(rets, ret_nodes): 
                    if ret is not np.nan:
                        node.backprop(ret)
                    
        # Update mcts_root root and select best action
        weights = [np.mean(child.results) for child in self.mcts_root.children]
        #print("Policy time: ", elapsed)
        self.mcts_root = self.mcts_root.children[np.argmax(weights)]
        return self.mcts_root.parent_action

    def act(self, observation):
        if self.method == "value_only":
            return self._value_only_policy(observation)
        elif self.method == "mcts":
            return self._mcts_policy(observation)
        else:
            raise ValueError(f"Method is unknown: {self.method}!!!")

    def update(self, old_observation, action, reward, observation):
        """
        If the policy uses MCTS need to update the tree with opponent's move.
        """
        if self.method == "mcts":
            # NOTE: This assumes the rl_env has get_last_move method
            opponent_move = self.rl_env.get_last_move()
            done = self.rl_env.is_done()
            # Check if move present in tree
            new_root = None
            for child in self.mcts_root.children:
                if child.parent_action == opponent_move:
                    new_root = child
            if not new_root:
                new_root = MCTSNode(value_fn=self.value_fn,
                                    parent_action=opponent_move,
                                    parent=self.mcts_root,
                                    terminal=done,
                                    exp_coef=self.mcts_root.exp_coef,)
            self.mcts_root = new_root
        

def get_rl_env(rl_env_name, render_mode):
    print(f"Creating environment: {rl_env_name}")
    print(f"crafter in {rl_env_name.lower()}: {'crafter' in rl_env_name.lower()}")
    if rl_env_name == "chess":
        from openelm.environments.rl_envs.chess_env import ChessEnv
        return ChessEnv(render_mode=render_mode)
    elif "crafter" in rl_env_name.lower():
        import gym as old_gym
        import crafter 
        return old_gym.make(rl_env_name)
    return get_wrapped_env(rl_env_name, render_mode)
    

class PolicyGenotype(Genotype):
    def __init__(self, 
                 policy_str: str,):
        """
        Genotype for policy parameterized by python coe.
        Args:
            policy_str: the policy in python code as a string
        """
        self.policy_str = policy_str

    def to_phenotype(self) -> Phenotype:
        raise ValueError("Undecided phenotype for python policies")


class ELMRLEnv(BaseEnvironment[PolicyGenotype]):
    def __init__(self,
                 config,
                 mutation_model,
                 render_mode=None,
                 num_procs=24,):
        """
        The objective is to generate well-performing python policies 
        for the given environment.
        + config: the config file path or dict
        + mutation_model: model used to mutate existing python policies
        + env: given RL environments
        + game_description: description of game
        + state_description: description of state
        """
        self.config = config
        self.batch_size = self.config.batch_size
        self.mutation_model = mutation_model
        self.task_type = TaskType.from_string(self.config.task_type)
        self.env = get_rl_env(self.config.rl_env_name, render_mode=render_mode)
        self.num_procs = num_procs
        self.api_list = self.config.api_list

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def _construct_prompt(self, exemplars: Optional[list[Program]] = None):
        prompt = get_task_prompt(self.task_type).format(**asdict(self.config))
        if exemplars is not None:
            demo = "Examples of policies: \n\n\n"
            for exemplar in exemplars:
                demo += f"```python\n{exemplar.src}```" + "\n\n"
                demo += f"The average return for the policy is: {exemplar.fitness}. You should optionally use this to determine how to improve/fix the policy.\n\n"
            prompt += demo

        return prompt

    def _extract_src_code(self, response: str):
        """Extracts the last code block from the agents response"""
        try:
            return re.findall(r"```python\n([^`]*)```", response)[-1]
        except IndexError:
            return ""

    def random(self) -> list[Program]:
        prompts = [{"prompt": self._construct_prompt()} for _ in range(self.config.batch_size)]
        responses = [self.mutation_model.generate_programs([prompt])[0] for prompt in prompts]
        new_programs = [Program(self._extract_src_code(response)) for response in responses]
        return new_programs
    
    def mutate(self, sol_list: List[List[Program]]) -> list[Program]:
        prompts = [{"prompt": self._construct_prompt(sols)} for sols in sol_list]
        responses = [self.mutation_model.generate_programs([prompt])[0] for prompt in prompts]
        new_programs = [Program(self._extract_src_code(response)) for response in responses]
        return new_programs
    
    def _extract_executable_policy(self, program: Program):
        if self.task_type == TaskType.POLICY:
            api_imports = ""
            for api in self.api_list:
                api_imports += f"from openelm.environments.rl_env_util.api_lib import {api}\n"
            source = f"{api_imports}{program.src}\n\npolicy = Policy()"
            result = dict()
            exec(source, result)
            policy_fn = result["policy"]
            return RLPolicy(self.env, policy_fn)
        elif self.task_type == TaskType.VALUE:
            source = f"{program.src}\n\nvalue = Value()"
            result = dict()
            exec(source, result)
            value_fn = result["value"]
            return RLValuePolicy(self.env, value_fn)

    def fitness(self, program: Program) -> dict:
        """
        Evaluate the proposed policy by rolling it out
        in the environment
        """
        env = self.env
        def eval_fitness(env, env_params, program, semaphore, returns, eval_runtimes, rank):
            semaphore.acquire()
            t = time()
            copy_env = env.deepcopy(mode="mp", **env_params) if hasattr(env, "deepcopy") else deepcopy(env)
            try:
                seed = np.random.randint(0, 1e9)
                observation, _ = copy_env.reset(seed=seed)
                policy = self._extract_executable_policy(program)
                rewards = []
                for _ in range(self.config.horizon):
                    action = policy.act(observation)
                    old_observation = observation
                    observation, reward, terminated, _, _ = copy_env.step(action)
                    policy.update(old_observation, action, reward, observation)
                    rewards.append(reward)
                    if terminated: break
                ret = reduce(lambda x, y: self.config.discount * x + y, rewards[::-1], 0)
            except Exception:
                ret = -100.0
            returns[rank] = float(ret)
            t = time() - t
            eval_runtimes[rank] = t
            semaphore.release()
        # Rollout policy in environment
        returns: List[float] = mp.Array("d", [0 for _ in range(self.config.fitness_curriculum.num_eval_rollouts)])
        eval_runtimes: List[float] = mp.Array("d", [0 for _ in range(self.config.fitness_curriculum.num_eval_rollouts)])
        semaphore = mp.Semaphore(self.num_procs)
        procs = []
        for rank in range(self.config.fitness_curriculum.num_eval_rollouts):
            env_params = self.config.fitness_curriculum.curriculum[rank]
            p = mp.Process(target=eval_fitness, args=(env, env_params, program, semaphore, returns, eval_runtimes, rank))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        returns = np.frombuffer(returns.get_obj(), dtype="d").tolist()
        eval_runtimes = np.frombuffer(eval_runtimes.get_obj(), dtype="d").tolist()
        fitness = np.mean(returns)
        res = dict(fitness=fitness, 
                   eval_runtimes=eval_runtimes,)
        return res


