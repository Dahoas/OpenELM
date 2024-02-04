import json
from abc import abstractmethod
import re
import warnings
from typing import Optional, List, Any
from functools import reduce
from copy import deepcopy
from enum import Enum
from dataclasses import asdict
from time import time

import numpy as np

from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.algorithms.fun_search import Program

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
        return ""
    elif task_type == TaskType.VALUE:
        return """\
You are responsible for designing a value function to solve the following task: 
{task_description}\n\n\
You will write a python `Value`, which should be initializable without any parameters from the user, object which has one method:
- `def value(observation)` which takes in an observation and returns the value of the observation
Note: You should not assume any exploration outside of what is learned during the agent's single rollout in \
the environment. This means you should not rely on Q-learning, requiring extra exploration.\n\n\
The observation space is defined formally as: 
{observation_description}\n\n\
You are allowed to use any python library you want but should not assume access \
to any other external resources (such as models with downloadable weights) unless otherwise specified. \
In particular you can assume access to the following APIs: \
{api_description}\n\n\
You should only write the Value class and nothing else. \
You are encouraged to be as creative as possible, do not simply copy one of the exemplars if given. \
All code should be written in a single, large code block.
"""


class RLPolicy:
    def __init__(self, 
                 rl_env,
                 policy,):
        self.rl_env = rl_env
        self.policy = policy

    def make_env_copy(self):
        return self.rl_env.deepcopy() if hasattr(self.rl_env, "deepcopy") else deepcopy(self.rl_env)

    def act(self, observation):
        return self.policy.act(observation)
    
    def update(self, old_observation, action, reward, observation):
        return self.policy.update(old_observation, action, reward, observation)


class RLValuePolicy(RLPolicy):
    """
    Implements methods of extracting policy from value function
    """
    def __init__(self,
                 rl_env,
                 value_fn,
                 method: str="value_only", 
                 depth: int=0, 
                 time_limit: float=0.5):
        self.rl_env = rl_env
        self.value_fn = value_fn
        self.method = method
        self.depth = depth
        self.time_limit = time_limit

    def _value_only_policy(self, observation):
        action_dict = {action: 0 for action in self.rl_env.get_actions()}
        for action in action_dict:
            rl_env_copy = self.make_env_copy()
            observation, reward, terminated, _, _ = rl_env_copy.step(action)
            value = self.value_fn.value(observation)
            action_dict[action] = reward if reward != 0 else value
        action_ind = np.argmax(list(action_dict.values()))

        return list(action_dict.keys())[action_ind]

    def act(self, observation):
        if self.method == "value_only":
            return self._value_only_policy(observation)
        else:
            raise ValueError(f"Method is unknown: {self.method}!!!")
        
    def update(self, old_observation, action, reward, observation):
        pass
    

def get_rl_env(rl_env_name, render_mode):
    if rl_env_name == "chess":
        from openelm.environments.chess_env import ChessEnv
        return ChessEnv(render_mode=render_mode)
    return gym.make(rl_env_name, render_mode=render_mode)
    

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
                 render_mode=None,):
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
                demo += f"```python\n{exemplar.src}```" + "\n\n\n"
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
            source = f"{program.src}\n\npolicy = Policy()"
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
        # Rollout policy in environment
        returns: List[float] = []
        eval_runtimes: List[float] = []
        trajectories: List[List[Any]] = []
        for _ in range(self.config.num_eval_rollouts):
            t = time()
            trajectory = []
            try:
                seed = np.random.randint(0, 1e9)
                observation, _ = env.reset(seed=seed)
                policy = self._extract_executable_policy(program)
                rewards = []
                for _ in range(self.config.horizon):
                    action = policy.act(observation)
                    trajectory.append(action)
                    old_observation = observation
                    observation, reward, terminated, _, _ = env.step(action)
                    policy.update(old_observation, action, reward, observation)
                    rewards.append(reward)
                    if terminated: break
                ret = reduce(lambda x, y: self.config.discount * x + y, rewards[::-1], 0)
            except Exception:
                ret = -100
            returns.append(ret)
            t = time() - t
            eval_runtimes.append(t)
            trajectories.append(trajectory)
        fitness = np.mean(returns)
        res = dict(fitness=fitness, 
                   eval_runtimes=eval_runtimes,
                   trajectories=trajectory,)
        return res