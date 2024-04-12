import numpy as np
from typing import List, Optional
from collections import defaultdict
from copy import deepcopy
import json

from openelm.environments.rl_env_util.utils import recursive_convert_lists_to_nparry, recursive_avg
from openelm.environments.rl_env_util.report_library import synthetic_reports


######## Core reports (computed every time) ########

def action_counts(observation, action, next_observation, memory):
    if action in memory:
        memory[action] += 1
    else:
        memory[action] = 1
    try:
        memory.pop("description")
        memory.pop("result")
    except KeyError:
        pass
    memory["result"] = deepcopy(memory)
    memory["description"] = "Average action counts"
    return memory


def get_min_max_return_actions(trajectories):
    worst_return = np.inf
    best_return = -np.inf
    for trajectory in trajectories:
        cur_return = np.sum(trajectory["rewards"])
        if cur_return > best_return:
            best_return = cur_return
            best_actions = trajectory["actions"]
        if cur_return < worst_return:
            worst_return = cur_return
            worst_actions = trajectory["actions"]
    return {"actions": worst_actions, "return": worst_return} , {"actions": best_actions, "return": best_return}


def num_trajectories(trajectories):
    return {"result": len(trajectories), 
            "description": "Number of evaluation rollouts."}


def avg_trajectory_reward(trajectories):
    return {"result": np.mean([sum(t["rewards"]) for t in trajectories]), 
            "description": "Average reward across trajectories."}


def avg_trajectory_length(trajectories):
    return {"result": np.mean([len(trajectory["actions"]) for trajectory in trajectories]),
            "description": "Average trajectory length.",}


core_reports = {
    "action_counts": action_counts,
    "get_min_max_return_actions": get_min_max_return_actions,
    "avg_trajectory_reward": avg_trajectory_reward,
    "avg_trajectory_length": avg_trajectory_length,
}


requires_full_trajectory = {  # List of reports which cannot be computed at the transition level
    "get_min_max_return_actions": get_min_max_return_actions,
    "num_trajectories": num_trajectories,
    "avg_trajectory_reward": avg_trajectory_reward,
    "avg_trajectory_length": avg_trajectory_length,
}


######## Compute metrics and helpers ########

def summarize_available_metrics():
    trajectory_path = "logs/trajectories/policy_2.json"
    from openelm.environments.rl_env_util.reports import compute_metrics as cm
    with open(trajectory_path, "r") as f:
        trajectories = json.load(f)
    metrics = compute_reports(trajectories)
    for metric in metrics.values():
        metric.pop("result")
    print(json.dumps(metrics, indent=2))


def compute_reports(trajectories: List[dict], extra_reports: dict={}):
    report_results = dict()
    report_list = {**core_reports, **extra_reports}

    # Preprocess trajectories
    for trajectory in trajectories:
        trajectory["observations"] = recursive_convert_lists_to_nparry(trajectory["observations"])
        trajectory["actions"] = recursive_convert_lists_to_nparry(trajectory["actions"])
        trajectory["rewards"] = recursive_convert_lists_to_nparry(trajectory["rewards"])

    for metric_name, metric in report_list.items():
        try:
            if metric_name in requires_full_trajectory:
                # Case 1: the report cannot be computed at the transition level
                report_result = metric(trajectories)
            else:
                # Case 2: the report can be computed at the transition level
                trajectory_results = []
                for trajectory in trajectories:
                    memory = dict(final_step=False)
                    observations, actions, rewards = trajectory["observations"], trajectory["actions"], trajectory["rewards"]
                    transition_results = []
                    for i in range(len(actions)):
                        if i == len(actions) - 1:
                            memory["final_step"] = True
                        obs = observations[i]
                        action = actions[i]
                        reward = rewards[i]
                        next_obs = observations[i+1]
                        report = metric(obs, action, next_obs, memory)
                    # Aggregate transition level metrics by summing
                    description = report["description"]
                    trajectory_result = report.pop("result")
                    trajectory_results.append(trajectory_result)
                report_result = recursive_avg(trajectory_results)
            report_results[metric_name] = {"result": report_result, "description": description}
        except InterruptedError:
            continue
    return report_results


if __name__ == "__main__":
    summarize_available_metrics()