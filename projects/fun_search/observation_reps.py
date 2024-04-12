
import json


def compress_observation_1(observation):
    """
    Compresses an observation into a symbolic representation.

    Parameters:
    - observation: The full observation from the environment.

    Returns:
    - A dictionary containing the compressed symbolic representation of the observation.
    """
    compressed = {}

    # Extract the direction the agent is facing.
    compressed['direction'] = observation['agent']['direction']
    
    # Compress the 7x7 image into counts of relevant objects in the agent's view.
    objects = ['unseen', 'empty', 'wall', 'floor', 'door', 'key', 'ball', 'box', 'goal', 'lava']
    colors = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
    states = ['door_open', 'door_closed', 'door_locked']
    
    image = observation['agent']['image']
    object_counts = {obj: 0 for obj in objects}
    color_counts = {color: 0 for color in colors}
    state_counts = {state: 0 for state in states}

    for row in image:
        for tile in row:
            object_counts[objects[tile[0]]] += 1
            color_counts[colors[tile[1]]] += 1
            if tile[0] == 4:  # If the object is a door, track its state.
                state_counts[states[tile[2]]] += 1

    compressed['objects'] = object_counts
    compressed['colors'] = color_counts
    compressed['states'] = state_counts

    # Compress the inventory into a simple list of objects being held.
    compressed['inventory'] = [objects[obj] for obj in observation['inv']]
    
    return compressed


def get_compressed_trajectories(trajectories: list) -> list:
    for trajectory in trajectories:
        trajectory["observations"] = [compress_observation_1(obs) for obs in trajectory["observations"]]
    return trajectories


def serialize_trajectories(trajectories: list) -> list:
    serialized_trajectories = []
    for trajectory in trajectories:
        serialized_trajectory = []
        for obs, action, reward in zip(trajectory["observations"][:-1], trajectory["actions"], trajectory["rewards"]):
            serialized_trajectory.append((obs, action, reward))
        serialized_trajectory.append((trajectory["observations"][-1], None, None))
        serialized_trajectories.append(serialized_trajectory)
    return serialized_trajectories


if __name__ == "__main__":
    trajectory_path = "logs/trajectories/policy_2.json"
    with open(trajectory_path, "r") as f:
        trajectories = json.load(f)
    trajectory_limit = 1
    trajectories = trajectories[:trajectory_limit]
    compressed_trajectories = get_compressed_trajectories(trajectories)
    serialized_compressed_trajectories = serialize_trajectories(compressed_trajectories)
    print(json.dumps(serialized_compressed_trajectories))

    