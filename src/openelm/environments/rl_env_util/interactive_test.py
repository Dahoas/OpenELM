import json
import argparse


from openelm.environments.rl_env_util.reports import compute_metrics as cm
from openelm.environments.rl_env_util.world_model import get_world_model, get_world_model_feedback


def compute_metrics():
    with open(trajectory_path, "r") as f:
        trajectories = json.load(f)
    metrics = cm(trajectories)
    print(json.dumps(metrics))


def compute_world_model_feedback():
    limit = 2
    with open(trajectory_path, "r") as f:
        trajectories = json.load(f)
    results = get_world_model_feedback(world_model, trajectories, action_model)[:limit]
    for result in results:
        print_dict(result)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--mode", choices=["compute_metrics",
                                           "compute_world_model",])
    args = parser.parse_args()

    if args.mode == "compute_metrics":
        func = compute_metrics
    elif args.mode == "compute_world_model":
        func = compute_world_model_feedback
        
    # Initialize global variables    
    trajectory_path = "logs/trajectories/policy_2.json"
    world_model_file = "init_policies/door_key/offline_analysis/world_model_1.py"
    action_model = [3, 4]
    world_model = get_world_model(world_model_file)

