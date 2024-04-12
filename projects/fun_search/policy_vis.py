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

from openelm.environments.rl_env_util.world_model import get_world_model_feedback, get_world_model
from utils import print_dict


# Execution loop
def execute():
    time_per_action = 1e-1 # Time between visualized moves in seconds
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
    
    print(hasattr(policy, "policy"))
    print(hasattr(policy.policy, "prepare_report"))
    if hasattr(policy, "policy") and hasattr(policy.policy, "prepare_report"):
        stats = policy.policy.report
        print(stats)
        report = policy.policy.prepare_report([stats])
        with open("report.txt", "w") as f:
            f.write(report)


def evaluate():
    res = elm_env.fitness(src)
    print(json.dumps(res, indent=2))
    with open("report.json", "w") as f:
        json.dump(res, f, indent=2)

        

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mode", choices=["execute", 
                                           "evaluate", 
                                           "compute_metrics",
                                           "compute_world_model",])
    args = parser.parse_args()

    if args.mode == "execute":
        func = execute
    elif args.mode == "evaluate":
        func = evaluate
    elif args.mode == "compute_metrics":
        func = compute_metrics
    elif args.mode == "compute_world_model":
        func = compute_world_model_feedback
        
    rl_env_name = "MiniGrid-UnlockPickup-v0-wrapped"
    task_type = "policy"
    curriculum = [{"stockfish_depth": i} for i in range(1, 21)]
    fitness_curriculum = FitnessCurriculum(num_eval_rollouts=20, curriculum=curriculum)
    horizon = 300  # 300
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
    render_mode = "human" if args.mode == "execute" else None
    elm_env = ELMRLEnv(config=config,
                    mutation_model=None,
                    render_mode=render_mode,)
    rl_env = elm_env.env

    # Set program
    policy_file = "init_policies/door_key/offline_analysis/policy_2.py"#human_policy.py"#report_design/policy_3_gpt4.py" #1_0.3.py"
    trajectory_path = "logs/trajectories/policy_2.json"
    world_model_file = "init_policies/door_key/offline_analysis/world_model_1.py"
    action_model = [3, 4]
    world_model = get_world_model(world_model_file)
    with open(policy_file, "r") as f:
        src = f.readlines()
        src = "\n".join(src)
    policy = elm_env._extract_executable_policy(program=src)

    if args.profile:
        cProfile.run(f"{func.__name__}()")
    else:
        func()

