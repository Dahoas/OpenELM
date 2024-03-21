import gymnasium as gym
import numpy as np
import time
from functools import reduce
from typing import Optional, List, Any
import json
import cProfile
import argparse

from openelm.environments.rl_env import ELMRLEnv, Program
from openelm.configs import RLEnvConfig, FitnessCurriculum
from rl_env_descriptions import envs


rl_env_name = "CrafterReward-v1"
rl_env_name_t = rl_env_name
task_type = "policy"
curriculum = [{"stockfish_depth": i} for i in range(1, 21)]
fitness_curriculum = FitnessCurriculum(num_eval_rollouts=20, curriculum=curriculum)
config = RLEnvConfig(rl_env_name=rl_env_name_t,
                             task_type="policy",
                             task_description=envs[rl_env_name]["task_description"],
                             observation_description=envs[rl_env_name]["observation_description"],
                             action_description=envs[rl_env_name]["action_description"],
                             reward_description=envs[rl_env_name]["reward_description"],
                             action_exemplar=envs[rl_env_name]["action_exemplar"],
                             fitness_curriculum=fitness_curriculum,
                             api_description="",)
                             #api_list=[],)
elm_env = ELMRLEnv(config=config,
                   mutation_model=None,
                   render_mode="human",)

prompt = elm_env._construct_prompt()
print("Prompt: ", prompt)
with open("dump.txt", "w") as f:
    f.write(prompt)
