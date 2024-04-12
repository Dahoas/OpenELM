import json
import numpy as np
from typing import List


def evaluate_obs_prediction(gt_obs, pred_obs):
    if pred_obs is None:
        return 0
    else:
        return int(gt_obs == pred_obs)
    

def run_world_model(world_model, trajectories, action_model) -> List[dict]:
    # First format trajectories into (obs, action, next_observation) tuples
    # Discard action transitions not modeled by world model (actions not in 'action_model')
    transitions = []
    for trajectory in trajectories:
        obs, acts = trajectory["observations"], trajectory["actions"]
        for i in range(len(acts)):
            transition = (obs[i], acts[i], obs[i+1])
            if acts[i] in action_model:
                transitions.append(transition)
    # Get world models predictions
    print("Num transitions: ", len(transitions))
    results = []
    for transition in transitions:
        obs, act, next_obs = transition
        result = world_model.predict(obs, act)
        result = dict(
            obs=obs,
            act=act,
            next_obs=next_obs,
            pred_obs=result["observation"],
            valid=result["valid"],
        )
        results.append(result)
    return results
        

def evaluate_world_model(world_model, trajectories, action_model):
    results = run_world_model(world_model, trajectories, action_model)
    scores = []
    for result in results:
        score = evaluate_obs_prediction(result["next_obs"], result["pred_obs"])
        scores.append(score)
    result = np.mean(scores)
    return result


def get_world_model_feedback(world_model, trajectories, action_model):
    """
    Pick out invalid actions
    """
    saved_keys = ["obs", "act", "next_obs"]
    results = run_world_model(world_model, trajectories, action_model)
    results = [{k: v for k, v in result.items() if k in saved_keys} for result in results if not result["valid"]]
    return results

def get_world_model(world_model_file):
    world_model_file = "init_policies/door_key/offline_analysis/world_model_1.py"
    with open(world_model_file, "r") as f:
        src = f.readlines()
        src = "\n".join(src)
    result = dict()
    exec(f"{src}\n\nworld_model = WorldModel()", result)
    world_model = result["world_model"]
    return world_model


if __name__ == "__main__":
    trajectory_path = "logs/trajectories/policy_2.json"
    from openelm.environments.rl_env_util.reports import compute_metrics as cm
    with open(trajectory_path, "r") as f:
        trajectories = json.load(f)
        
    world_model_file = "init_policies/door_key/offline_analysis/world_model_1.py"
    world_model = get_world_model(world_model_file)
    action_model = [3, 4]
    result = evaluate_world_model(world_model, trajectories, action_model)
    print("Accuracy: ", result)