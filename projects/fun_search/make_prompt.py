import gymnasium as gym
import numpy as np
import time
from functools import reduce
from typing import Optional, List, Any
import json
import cProfile
import argparse

from rl_env_descriptions import envs
from openelm.environments.rl_env_util.prompts import prompts


policy_file = "/home/tsawada/Files/projects/o2401/OpenELM/projects/fun_search/seed_policies/gpt4/2403210015.py"


with open(policy_file, "r") as f:
    src = f.readlines()
    src = "\n".join(src)

rl_env_name = "CrafterReward-v1"
prompt = ""
prompt += prompts["env_prompt"].format(**envs[rl_env_name])
prompt += prompts["policy_prompt"].format(policy=src, ret=0.4, rollouts=10, max_steps=1000)
prompt += prompts["metrics_prompt"]

with open("prompt.txt", "w") as f:
    f.write(prompt)