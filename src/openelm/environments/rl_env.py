import json
import re
import warnings
from typing import Optional, List
from functools import reduce

import numpy as np

from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.algorithms.fun_search import Program

import gymnasium as gym


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
                 mutation_model,):
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
        
        self.env = gym.make(self.config.rl_env_name)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def _construct_prompt(self, exemplars: Optional[list[Program]] = None):
        prompt = f"""\
You are responsible for designing a decision policy to solve the following task: 
{self.config.task_description}\n\n\
You will write a python `Policy()`, which should be initializable without any parameters from the user, object which has two methods:
- `def act(observation)` which takes in an observation and returns an action.
- `update(observation, action, reward, next_observation)` which takes in the current observation, \
chosen action, reward, and next_observation and updates any persistent memory/state between observations. \
Note: You should not assume any exploration outside of what is learned during the agent's single rollout in \
the environment. This means you should not rely on Q-learning, etc.\n\n\
The observation space is defined formally as: 
{self.config.observation_description}\n\n\
The action space is defined formally as:
{self.config.action_description}\n\n\
The rewards are defined formally as:
{self.config.reward_description}\n\n\
Consider the following example action sequence to familiairize yourself with the env dynamics\n\n\
{self.config.action_exemplar}\n\n\
You are allowed to use any python library you want but should not assume access \
to any other external resources (such as models with downloadable weights) unless otherwise specified. \
In particular you can assume access to the following APIs: \
{self.config.api_description}\n\n\
You should only write the Policy class and nothing else. \
You are encouraged to be as creative as possible, do not simply copy one of the exemplars if given. \
All code should be written in a single, large code block.
"""
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
        source = f"{program.src}\n\npolicy = Policy()"
        result = dict()
        exec(source, result)
        return result["policy"]

    def fitness(self, program: Program) -> float:
        """
        Evaluate the proposed policy by rolling it out
        in the environment
        """
        env = self.env
        # Rollout policy in environment
        returns = []
        for _ in range(self.config.num_eval_rollouts):
            # First extract executable policy from source
            try:
                policy = self._extract_executable_policy(program)
                seed = np.random.randint(0, 1e9)
                observation, _ = env.reset(seed=seed)
                rewards = []
                for _ in range(self.config.horizon):
                    action = policy.act(observation)
                    old_observation = observation
                    observation, reward, terminated, _, _ = env.step(action)
                    policy.update(old_observation, action, reward, observation)
                    rewards.append(reward)
                    if terminated: break
                ret = reduce(lambda x, y: self.config.discount * x + y, rewards[::-1], 0)
            except Exception:
                ret = -100
            returns.append(ret)
        fitness = np.mean(returns)
        return fitness