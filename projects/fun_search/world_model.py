import json
import numpy as np


def evaluate_obs_prediction(gt_obs, pred_obs):
    if pred_obs is None:
        return 0
    else:
        return int(gt_obs == pred_obs)


def evaluate_world_model(world_model, trajectories, action_model):
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
    scores = []
    for transition in transitions:
        obs, act, next_obs = transition
        #try:
        pred_obs = world_model.predict(obs, act)
        #except Exception:
        #    pred_obs = None
        score = evaluate_obs_prediction(next_obs, obs)
        scores.append(score)
    result = np.mean(scores)
    return result


if __name__ == "__main__":
    trajectory_path = "logs/trajectories/policy_2.json"
    from metrics import compute_metrics as cm
    with open(trajectory_path, "r") as f:
        trajectories = json.load(f)
        
    world_model_file = "init_policies/door_key/offline_analysis/world_model_1.py"
    with open(world_model_file, "r") as f:
        src = f.readlines()
        src = "\n".join(src)
    result = dict()
    exec(f"{src}\n\nworld_model = WorldModel()", result)
    world_model = result["world_model"]
    action_model = [3, 4]
    result = evaluate_world_model(world_model, trajectories, action_model)
    print("Accuracy: ", result)