import json
from abc import abstractmethod
import re
import warnings
from typing import Optional, List, Any
from functools import reduce
from copy import deepcopy
from enum import Enum
from dataclasses import asdict, dataclass
import time as time_t
from time import time
import multiprocessing as mp
import os
import pathlib
import uuid

import numpy as np

from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.environments.rl_envs.mcts import MCTSNode
from openelm.algorithms.fun_search import Program
from openelm.environments.rl_env_util.env_wrappers import get_wrapped_env
from openelm.environments.utils import robust_dump_json
from openelm.mutation_models.rl_mutation_model import PromptMode, MutationMode

import gymnasium as gym


EVAL_TIMEOUT = 60


class TaskType(Enum):
    POLICY = 1

    @classmethod
    def from_string(cls, name):
        name_to_val = {val.name: val for val in cls}
        if name_to_val.get(name.upper(), None):
            return name_to_val[name.upper()]
        else:
            raise ValueError(f"Unknown name: {name}!!!")


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
        

def get_rl_env(rl_env_name, render_mode):
    if rl_env_name == "chess":
        from openelm.environments.rl_envs.chess_env import ChessEnv
        return ChessEnv(render_mode=render_mode)
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
        self.trajectories_dir = os.path.join(config.output_dir, "trajectories")
        self.scratch_dir = os.path.join(config.output_dir, "scratch")
        self.env_dict = {
            "task_description": self.config.task_description,
            "observation_description": self.config.observation_description,
            "action_description": self.config.action_description,
            "reward_description": self.config.reward_description,
        }
        
        # Make env reports folder in output dir
        path = pathlib.Path(self.trajectories_dir)
        path.mkdir(exist_ok=True, parents=True)
        path = pathlib.Path(self.scratch_dir)
        path.mkdir(exist_ok=True, parents=True)

    def random(self) -> List[dict]:
        batch = [self.env_dict for _ in range(self.config.batch_size)]
        outputs = self.mutation_model(batch, 
                                       prompt_mode=PromptMode.DESIGNER, 
                                       mutation_mode=MutationMode.UNCONDITIONAL)
        return outputs
    
    def mutate(self, 
               program_list: List[List[Program]],  # In practice this will always be one program per inner list
               prompt_mode: PromptMode,
               mutation_mode: MutationMode) -> List[dict]:
        # Flatten
        program_list = [l[0] for l in program_list]
        batch = [{**self.env_dict, **{
            "policy": program.src,
            "critique": program.critique,
            "trajectory_path": program.trajectory_path,
        }} for program in program_list]
        outputs = self.mutation_model(batch, 
                                       prompt_mode=prompt_mode, 
                                       mutation_mode=mutation_mode)
        return outputs
    
    def _extract_executable_policy(self, program: str):
        if self.task_type == TaskType.POLICY:
            api_imports = ""
            for api in self.api_list:
                api_imports += f"from openelm.environments.rl_env_util.api_lib import {api}\n"
            source = f"{api_imports}{program}\n\npolicy = Policy()"
            result = dict()
            exec(source, result)
            policy_fn = result["policy"]
            return RLPolicy(self.env, policy_fn)

    def fitness(self, src: str) -> dict:
        """
        Evaluate the proposed policy by rolling it out
        in the environment
        """
        env = self.env
        def eval_fitness(env, env_params, program, seed, semaphore, returns, eval_runtimes, rank):
            semaphore.acquire()
            t = time()
            copy_env = env.deepcopy(mode="mp", **env_params) if hasattr(env, "deepcopy") else deepcopy(env)
            try:
                observations, actions, rewards = [], [], []
                observation, _ = copy_env.reset(seed=seed)
                observations.append(observation)
                policy = self._extract_executable_policy(program)
                for _ in range(self.config.horizon):
                    action = policy.act(observation)
                    old_observation = observation
                    observation, reward, terminated, _, _ = copy_env.step(action)
                    policy.update(old_observation, action, reward, observation)
                    rewards.append(reward)
                    actions.append(action)
                    observations.append(observation)
                    if terminated: break
                ret = reduce(lambda x, y: self.config.discount * x + y, rewards[::-1], 0)
                trajectory = {"observations": observations, "actions": actions, "rewards": rewards}
                # Log trajectory data
                report_file = os.path.join(self.scratch_dir, f"{seed}.json")
                robust_dump_json(trajectory, report_file)
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
            seed = np.random.randint(0, 1e9)
            p = mp.Process(target=eval_fitness, args=(env, env_params, src, seed, semaphore, returns, eval_runtimes, rank))
            p.start()
            procs.append(p)
        
        def join_under_timeout(procs, time_limit):
            t = time()
            elapsed = time() - t
            procs_done: list[bool] = len(procs)*[False]
            while sum(procs_done) < len(procs) or elapsed > time_limit:
                for i, p in enumerate(procs):
                    # Check if process has finished running
                    if not p.is_alive():
                        procs_done[i] = True
                        p.join()
                time_t.sleep(1)
                elapsed = time() - t
            for p in procs:
                p.terminate()
                p.join()
        join_under_timeout(procs, time_limit=EVAL_TIMEOUT)

        returns = np.frombuffer(returns.get_obj(), dtype="d").tolist()
        eval_runtimes = np.frombuffer(eval_runtimes.get_obj(), dtype="d").tolist()
        fitness = np.mean(returns)
        # Recover trajectory data
        trajectory_files = pathlib.Path(self.scratch_dir).glob("*.json")
        trajectories = []
        for trajectory in trajectory_files:
            with open(trajectory, "r") as f:
                trajectory = json.load(f)
            trajectories.append(trajectory)
        # Save trajectory data to file
        trajectory_path = os.path.join(self.trajectories_dir, f"{str(uuid.uuid4())}.json")
        robust_dump_json(trajectories, trajectory_path)
        res = dict(fitness=fitness, 
                   eval_runtimes=eval_runtimes,
                   trajectory_path=trajectory_path,)
        # Clear scratch directory 
        # NOTE: assumes fitness evaluation is single-threaded after this line
        trajectory_files = pathlib.Path(self.scratch_dir).glob("*.json")
        for trajectory in trajectory_files:
            trajectory.unlink()
        return res

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")