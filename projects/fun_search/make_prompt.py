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


policy_file = "init_policies/door_key/offline_analysis/policy_1.py"#human_policy.py"#report_design/policy_3_gpt4.py" #1_0.3.py"
with open(policy_file, "r") as f:
    src = f.readlines()
    src = "\n".join(src)

rl_env_name = "MiniGrid-UnlockPickup-v0"
prompt = ""
prompt += prompts["env_prompt"].format(**envs[rl_env_name])
prompt += prompts["policy_prompt"].format(policy=src, ret=0.4)
prompt += prompts["metrics_prompt"]

with open("prompt.txt", "w") as f:
    f.write(prompt)