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

import numpy as np

from openelm.environments.base import BaseEnvironment, Genotype, Phenotype
from openelm.environments.rl_envs.mcts import MCTSNode
from openelm.algorithms.fun_search import Program
from openelm.environments.rl_env_util.env_wrappers import get_wrapped_env

import gymnasium as gym


EVAL_TIMEOUT = 60


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
        is_complete_keyword = "<DONE>"
        return """\
You are responsible for designing a decision policy to solve the following task: 
{{task_description}}\n\n\
You will write a python `Policy()`, which should be initializable without any parameters from the user, object which has two methods:
- `def act(observation)` which takes in an observation and returns an action.
- `update(observation, action, reward, next_observation)` which takes in the current observation, \
chosen action, reward, and next_observation and updates any persistent memory/state between observations. \
You should never assume the actions you take. `update` is a good place to test your understanding of the world and record the results.
- `report: dict` which collects observations and statistics from over the agent's execution to understand its performance in the environement.
This is also a good way to test your own understanding of the environment's dynamics. This should be serializable. \
Each key should have the form 'name'/'aggregation' where 'name' is the metric name and 'aggregation' determines how the 
statistic is aggregated over multiple runs. 'aggregation' should be one of ['mean', 'max', 'min']
Note: You should not assume any exploration outside of what is learned during the agent's single rollout in \
the environment. This means you should not rely on Q-learning, etc.\n\n\
The observation space is defined formally as: 
{{observation_description}}\n\n\
The action space is defined formally as:
{{action_description}}\n\n\
The rewards are defined formally as:
{{reward_description}}\n\n\
Consider the following example action sequence to familiairize yourself with the env dynamics\n\n\
{{action_exemplar}}\n\n\
You are allowed to use any python library you want but should not assume access \
to any other external resources (such as models with downloadable weights) unless otherwise specified. \
In particular you can assume access to the following APIs: \
{{api_description}}\n\n\
You should only write the Policy class and nothing else. \
You are encouraged to be as creative as possible, do not simply copy one of the exemplars if given. \
Your policy should also be robust to adversity. If it finds itself getting stuck, repeating the same moves, \
it should try something new. Exploration is encouraged, but keep in mind the goal and the action preconditions.\
Make sure when you are engaging in exploration, it is diverse and incentivizing discovery of new states.\n\
Thank carefully about the order of steps you propose. Executing the right steps in the wrong order will be costly\n\
All code should be written in a single, large code block. \
When you are finished with your response you should write {is_complete_keyword} at the very end outside any code blocks.
""".format(is_complete_keyword=is_complete_keyword)
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
All code should be written in a single, large code block. \
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
    
    def prepare_trajectories(self, trajectories) -> str:
        """
        Summarizes trajectories generated by current policy for evaluation with LLM.
        """
        return ""
    
    def prepare_report(self, reports):
        if len(reports) == 0: return {}
        trajectories = [report["trajectory"] for report in reports]
        reports = [report["report"] for report in reports]
        agg_dict = {"mean": np.mean, "max": np.max, "min": np.min}
        final_report = ""
        for k in reports[0]:
            agg_name = k.split("/")[-1]
            agg = agg_dict.get(agg_name, np.mean)
            stat = agg([report[k] for report in reports])
            final_report += f"{k}: {stat}\n"
        final_report += f"num_policy_runs: {len(reports)}"
        # Now process trajectories
        final_report += self.prepare_trajectories(trajectories)
        return final_report


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
    
    
class PromptMode(Enum):
    DEFAULT = 1
    FEEDBACK = 2

    @classmethod
    def from_string(cls, name):
        name_to_val = {val.name: val for val in cls}
        if name_to_val.get(name.upper(), None):
            return name_to_val[name.upper()]
        else:
            raise ValueError(f"Unknown name: {name}!!!")


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
        self.reports_dir = os.path.join(config.output_dir, "reports")
        
        # Make env reports folder in output dir
        path = pathlib.Path(self.reports_dir)
        path.mkdir(exist_ok=True, parents=True)

    def get_rng_state(self) -> Optional[np.random._generator.Generator]:
        warnings.warn("WARNING: rng state not used in this environment")
        return None

    def set_rng_state(self, rng_state: Optional[np.random._generator.Generator]):
        warnings.warn("WARNING: rng state not used in this environment")
        pass

    def _construct_prompt(self, exemplars: Optional[list[Program]] = None):
        # Sample prompting mode
        p = np.ones(len(PromptMode)) / len(PromptMode)
        mode = np.random.choice(list(PromptMode), p=p)
        prompt = get_task_prompt(self.task_type).format(**asdict(self.config))
        if exemplars is not None:
            demo = "Examples of policies: \n\n\n"
            for exemplar in exemplars:
                demo += f"```python\n{exemplar.src}```" + "\n\n"
                demo += f"Average Policy Return:\n{exemplar.fitness}\n"
                if mode == PromptMode.FEEDBACK:
                    demo += f"Policy Report: \n{exemplar.report}\n"
                    demo += "Before writing any code think about what you can learn from the policy report. Focus on finding and fixing one thing at a time. This may include proposing \
    additional reports to better understand the policy's behavior."
            prompt += demo
        return prompt

    def _extract_src_code(self, response: str):
        """Extracts the last code block from the agents response"""
        is_complete_keyword = "<DONE>"
        try:
            response = response.replace(is_complete_keyword, "")
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
                # Log report to report dir
                if hasattr(policy, "policy") and hasattr(policy.policy, "report"):
                    report = policy.policy.report
                    env_feedback = {"trajectory": trajectory, "report": report}
                    report_file = os.path.join(self.reports_dir, f"{seed}.json")
                    with open(report_file, "w") as f:
                        json.dump(env_feedback, f)
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
            p = mp.Process(target=eval_fitness, args=(env, env_params, program, seed, semaphore, returns, eval_runtimes, rank))
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
        # Recover report
        try:
            reports = pathlib.Path(self.reports_dir).glob("*.json")
            reports_dicts = []
            for report in reports:
                with open(report, "r") as f:
                    report = json.load(f)
                reports_dicts.append(report)
            policy = self._extract_executable_policy(program)
            report = policy.prepare_report(reports_dicts)
        except Exception:
            report = ""
        res = dict(fitness=fitness, 
                   eval_runtimes=eval_runtimes,
                   report=report,)
        # Clear reports directory 
        # NOTE: assumes fitness evaluation is single-threaded after this line
        reports = pathlib.Path(self.reports_dir).glob("*.json")
        for report in reports:
            report.unlink()
        return res


