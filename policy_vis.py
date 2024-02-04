import gymnasium as gym
import numpy as np
import time
from functools import reduce
from typing import Optional, List, Any

from openelm.environments.rl_env import ELMRLEnv, Program
from openelm.configs import RLEnvConfig


rl_env_name = "chess"
task_type = "value"
config = RLEnvConfig(rl_env_name=rl_env_name,
                     task_type=task_type,
                     task_description="",
                     observation_description="",
                     action_description="",
                     reward_description="",
                     action_exemplar="",)
elm_env = ELMRLEnv(config=config,
                   mutation_model=None,
                   render_mode="human",)
rl_env = elm_env.env

# Set program
policy_file = "init_policies/chess/value_1.py"
with open(policy_file, "r") as f:
    src = f.readlines()
    src = "\n".join(src)
program = Program(src=src)
policy = elm_env._extract_executable_policy(program=program)

# Execution loop
time_per_action = 1  # Time between visualized moves in seconds
seed = np.random.randint(0, 1e9)
observation, _ = rl_env.reset(seed=seed)
rl_env.render()
rewards = []
for step in range(config.horizon):
    print("Step num: ", step)
    action = policy.act(observation)
    old_observation = observation
    observation, reward, terminated, _, _ = rl_env.step(action)
    rewards.append(reward)
    policy.update(old_observation, action, reward, observation)
    rl_env.render()
    time.sleep(time_per_action)
    if terminated: break
ret = reduce(lambda x, y: config.discount * x + y, rewards[::-1], 0)

