import gymnasium as gym
import numpy as np
import time
from functools import reduce
from typing import Optional, List, Any
import json
import cProfile
import argparse
from openelm.configs import RLEnvModelConfig

from openelm.environments.rl_env_util.rl_env_descriptions import envs
from openelm.mutation_models.prompts import designer_prompts
from openelm.mutation_models.rl_mutation_model import RLEnvModel, MutationMode


rl_env_name = "MiniGrid-UnlockPickup-v0"
env = envs[rl_env_name]

mutation_config = RLEnvModelConfig(model_type="gptquery",
                                    designer_model_path="gpt-3.5-turbo-0125", #"gpt-4-0125-preview",  # gpt-3.5-turbo-0125, # claude-3-haiku-20240307
                                    analyzer_model_path="gpt-4-turbo",
                                    designer_temp=1.1,
                                    analyzer_temp=0.3,
                                    gen_max_len=4096,
                                    batch_size=1,
                                    model_path="",)
mutation_config.logging_path = ""
model = RLEnvModel(mutation_config)

policy_file = "init_policies/door_key/offline_analysis/policy_1.py"#human_policy.py"#report_design/policy_3_gpt4.py" #1_0.3.py"
with open(policy_file, "r") as f:
    src = f.readlines()
    policy = "\n".join(src)

batch = [{**env, "policy": policy}]
mutation_mode = MutationMode.UNCONDITIONAL
prompt = model.policy_designer.construct_input(batch, mutation_mode=mutation_mode)[0]["prompt"]

with open("prompt.txt", "w") as f:
    f.write(prompt)