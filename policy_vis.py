import gymnasium as gym
import numpy as np
import time
from functools import reduce
from typing import Optional, List, Any
import json
import cProfile
import argparse

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
                     action_exemplar="",
                     horizon=100,)
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
def execute():
    time_per_action = 1e-5 # Time between visualized moves in seconds
    seed = np.random.randint(0, 1e9)
    observation, _ = rl_env.reset(seed=seed)
    rl_env.render()
    rewards = []
    for step in range(config.horizon):
        print("Step num: ", step)
        action = policy.act(observation)
        old_observation = observation
        observation, reward, terminated, _, info = rl_env.step(action)
        rewards.append(reward)
        policy.update(old_observation, action, reward, observation)
        if time_per_action > 1e-3:
            rl_env.render()
        if not time_per_action and input():
            continue
        else:
            time.sleep(time_per_action)
        if terminated: break
    print("Info: ")
    print(json.dumps(info, indent=2))
    ret = reduce(lambda x, y: config.discount * x + y, rewards[::-1], 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if args.profile:
        cProfile.run("execute()")
    else:
        execute()

