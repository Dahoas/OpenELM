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


rl_env_name = "MiniGrid-UnlockPickup-v0-wrapped"
task_type = "policy"
curriculum = [{"stockfish_depth": i} for i in range(1, 21)]
fitness_curriculum = FitnessCurriculum(num_eval_rollouts=20, curriculum=curriculum)
config = RLEnvConfig(rl_env_name=rl_env_name,
                     task_type=task_type,
                     task_description="",
                     observation_description="",
                     action_description="",
                     reward_description="",
                     action_exemplar="",
                     api_list=[],
                     horizon=horizon,
                     fitness_curriculum=fitness_curriculum,)
elm_env = ELMRLEnv(config=config,
                   mutation_model=None,
                   render_mode="human",)
rl_env = elm_env.env

# Set program
policy_file = "init_policies/door_key/policy_update_geminiA_3.py" #1_0.3.py"
with open(policy_file, "r") as f:
    src = f.readlines()
    src = "\n".join(src)
program = Program(src=src)
policy = elm_env._extract_executable_policy(program=program)

# Execution loop
def execute():
    time_per_action = 0.1 # Time between visualized moves in seconds
    seed = np.random.randint(0, 1e9)
    observation, _ = rl_env.reset(seed=seed)
    rl_env.render()
    rewards = []
    game_start = time.time()
    for step in range(config.horizon):
        print("Step num: ", step)
        t = time.time()
        action = policy.act(observation)
        elapsed = time.time() - t
        print("Policy time: ", elapsed)
        old_observation = observation
        t = time.time()
        observation, reward, terminated, _, info = rl_env.step(action)
        elapsed = time.time() - t
        print("Env time: ", elapsed)
        rewards.append(reward)
        if task_type == "value":
            value = policy.value_fn(observation)
            print("Value: ", value)
            print("MCTS Value: ", np.mean(policy.mcts_root.results))
        policy.update(old_observation, action, reward, observation)
        if time_per_action > 1e-3:
            print(action)
            rl_env.render()
        if not time_per_action and input():
            continue
        else:
            time.sleep(time_per_action)
        if terminated: break
    print("Info: ")
    print(json.dumps(info, indent=2))
    ret = reduce(lambda x, y: config.discount * x + y, rewards[::-1], 0)
    print("Reward: ", ret)
    print(rewards)
    game_time = time.time() - game_start
    print("Game time: ", game_time)


def evaluate():
    res = elm_env.fitness(program)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mode", choices=["execute", "evaluate"])
    args = parser.parse_args()

    if args.mode == "execute":
        func = execute
    else:
        func = evaluate

    if args.profile:
        cProfile.run(f"{func.__name__}()")
    else:
        func()

