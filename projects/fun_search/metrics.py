import numpy as np


agg_functions = {
    "min": np.min,
    "max": np.max,
    "mean": np.mean,
    "sum": np.sum,
}

########## Transition-level metrics ##########

def success_rate(observation, action, next_observation, memory):
    """
    Computes the success rate of picking up the box over a trajectory.
    """
    description = "Success rate of picking up the box"
    result = 1.0 if observation['inv'] == [7] else 0.0
    return {"result": result, "description": description}

def key_pickup_success(observation, action, next_observation, memory):
    if 'key_pickup_attempts' not in memory:
        memory['key_pickup_attempts'] = 0
        memory['key_pickups'] = 0
    if observation['agent']['image'][3][5][0] == 5 and action == 3:  # If facing a key and action is to pick up
        memory['key_pickup_attempts'] += 1
        if 5 in next_observation['inv']:  # If next observation has key in inventory
            memory['key_pickups'] += 1
    if memory['key_pickup_attempts'] > 0:
        success_rate = memory['key_pickups'] / memory['key_pickup_attempts']
    else:
        success_rate = 0.0
    return {"result": success_rate, "description": "Key pickup success rate"}

def door_interaction_efficiency(observation, action, next_observation, memory):
    if 'door_interactions' not in memory:
        memory['door_interactions'] = 0
        memory['successful_door_opens'] = 0
    if observation['agent']['image'][3][5][0] == 4 and action == 5:  # If facing a door and action is to toggle
        memory['door_interactions'] += 1
        if next_observation['agent']['image'][3][5][2] == 0:  # If next observation shows door is open
            memory['successful_door_opens'] += 1
    if memory['door_interactions'] > 0:
        efficiency = memory['successful_door_opens'] / memory['door_interactions']
    else:
        efficiency = 0.0
    return {"result": efficiency, "description": "Door interaction efficiency"}

def exploration_efficiency(observation, action, next_observation, memory):
    if 'moves' not in memory:
        memory['moves'] = 0
        memory['new_tiles_explored'] = 0
        memory['explored_tiles'] = set()
    memory['moves'] += 1
    forward_tile = tuple(observation['agent']['image'][3][5])
    if action == 2 and forward_tile not in memory['explored_tiles']:  # If the action is to move forward and the tile is new
        memory['new_tiles_explored'] += 1
        memory['explored_tiles'].add(forward_tile)
    if memory['moves'] > 0:
        efficiency = memory['new_tiles_explored'] / memory['moves']
    else:
        efficiency = 0.0
    return {"result": efficiency, "description": "Exploration efficiency"}

def report_box_pickup_attempts(observation, action, next_observation, memory):
    """
    Tracks the number of times the policy attempted to pick up the box.
    """
    if "box_pickup_attempts" not in memory:
        memory["box_pickup_attempts"] = 0
    if action == 3 and observation['agent']['image'][3][5][0] == 7:  # If action is pickup and facing a box
        memory["box_pickup_attempts"] += 1
    memory["description"] = "Number of attempts made to pick up the box."
    memory["result"] = memory.get("box_pickup_attempts", 0)
    return memory


def report_steps_before_box_pickup(observation, action, next_observation, memory):
    """
    Calculates the number of steps taken before the first box pickup attempt.
    """
    if "first_box_pickup_step" not in memory:
        memory["first_box_pickup_step"] = -1
        memory["steps"] = 0
    memory["steps"] += 1
    if action == 3 and observation['agent']['image'][3][5][0] == 7 and memory["first_box_pickup_step"] == -1:  # If action is pickup and facing a box
        memory["first_box_pickup_step"] = memory["steps"]
    memory["description"] = "Average number of steps taken before the first attempt to pick up the box."
    memory["result"] = memory.get("first_box_pickup_step", 0)
    return memory


def report_distance_traveled_before_box_pickup(observation, action, next_observation, memory):
    """
    Calculates the distance traveled before attempting to pick up the box.
    """
    if "distance_traveled" not in memory:
        memory["distance_traveled"] = 0
        memory["attempted_box_pickup"] = False
    if action == 2:  # If action is move forward
        memory["distance_traveled"] += 1
    if action == 3 and observation['agent']['image'][3][5][0] == 7:  # If action is pickup and facing a box
        memory["attempted_box_pickup"] = True
    memory["description"] = "Distance traveled before attempting to pick up the box."
    if memory["attempted_box_pickup"]:
        memory["result"] = memory.get("distance_traveled", 0)
    else:
        memory["result"] = 0
    return memory


def report_efficiency_of_movement(observation, action, next_observation, memory):
    """
    Calculates the ratio of successful movements to total movements.
    """
    if "successful_moves" not in memory:
        memory["successful_moves"] = 0
        memory["total_moves"] = 0
    if action == 2:  # If action is move forward
        memory["total_moves"] += 1
        if next_observation['agent']['image'][3][6][0] not in [2, 4, 9]:  # If next tile is not a wall, door, or lava
            memory["successful_moves"] += 1
    memory["description"] = "Efficiency of movement."
    if memory["total_moves"] > 0:
        memory["result"] = memory["successful_moves"] / memory["total_moves"]
    else:
        memory["result"] = 0
    return memory


########## Compute metric function ##########

metrics = {
    "success_rate": success_rate,
    "key_pickup_success": key_pickup_success,
    "door_interaction_efficiency": door_interaction_efficiency,
    "exploration_efficiency": exploration_efficiency,
    "report_box_pickup_attempts": report_box_pickup_attempts,
    "report_steps_before_box_pickup": report_steps_before_box_pickup,
    "report_distance_traveled_before_box_pickup": report_distance_traveled_before_box_pickup,
    "report_efficiency_of_movement": report_efficiency_of_movement,
}

metric_aggs = {

}

def compute_metrics(trajectories: list[dict]):
    metric_results = {
        "num_trajectories": {"result": len(trajectories), "description": "Number of evaluation rollouts."},
        "avg_trajectory_reward": {"result": np.mean([sum(t["rewards"]) for t in trajectories]), "description": "Average reward across trajectories."},
    }
    for metric_name, metric in metrics.items():
        print(metric_name)
        try:
            trajectory_results = []
            memory = dict()
            for trajectory in trajectories:   
                observations, actions, rewards = trajectory["observations"], trajectory["actions"], trajectory["rewards"]
                transition_results = []
                for i in range(len(actions)):
                    obs = observations[i]
                    action = actions[i]
                    reward = rewards[i]
                    next_obs = observations[i+1]
                    report = metric(obs, action, next_obs, memory)
                    description = report.pop("description")
                # Aggregate transition level metrics by summing
                trajectory_result = report.pop("result")
                trajectory_results.append(trajectory_result)
            metric_result = metric_aggs.get(metric_name, np.mean)(trajectory_results)
            metric_results[metric_name] = {"result": metric_result, "description": description}
        except InterruptedError:
            continue
    return metric_results
                
