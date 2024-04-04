def steps_to_key_pickup(observation, action, reward, next_observation, memory):
    if 'steps_to_key' not in memory:
        memory['steps_to_key'] = 0
        memory['key_picked'] = False

    if not memory['key_picked']:
        memory['steps_to_key'] += 1

    if reward == 0.1 and not memory['key_picked']:  # Assuming reward of 0.1 is for key pickup
        memory['key_picked'] = True
    
    return {"result": memory['steps_to_key'] if memory['key_picked'] else 0, "description": "Steps taken to pick up the key."}

def door_interaction_efficiency(observation, action, reward, next_observation, memory):
    if 'key_picked' not in memory:
        memory['key_picked'] = False
        memory['door_opened'] = False
        memory['actions_after_key'] = 0

    if memory['key_picked'] and not memory['door_opened']:
        memory['actions_after_key'] += 1

    if reward == 0.1:  # Key pickup
        memory['key_picked'] = True

    if reward == 0.2:  # Door opening
        memory['door_opened'] = True
        result = memory['actions_after_key']
    else:
        result = 0

    return {"result": result, "description": "Actions taken between key pickup and door opening."}

def exploration_efficiency(observation, action, reward, next_observation, memory):
    if 'explored_tiles' not in memory:
        memory['explored_tiles'] = set()
        memory['key_picked'] = False

    current_position = (observation['agent']['image'][3][6][1], observation['agent']['image'][3][6][2])  # Unique identifier for current tile based on color and state
    memory['explored_tiles'].add(current_position)

    if reward == 0.1:  # Key pickup
        memory['key_picked'] = True

    return {"result": len(memory['explored_tiles']) if not memory['key_picked'] else 0, "description": "Unique tiles explored before key pickup."}

def successful_box_pickup(observation, action, reward, next_observation, memory):
    if 'box_picked' not in memory:
        memory['box_picked'] = False

    if reward == 1 - 0.9 * (memory.get('steps', 0) / 300):  # Assuming this reward is for successful box pickup
        memory['box_picked'] = True

    return {"result": 1 if memory['box_picked'] else 0, "description": "Whether the box was successfully picked up."}

def path_efficiency_to_key(observation, action, reward, next_observation, memory):
    if 'key_picked_up' not in memory:
        memory['key_picked_up'] = False
        memory['steps_to_key'] = 0
    if not memory['key_picked_up']:
        memory['steps_to_key'] += 1
    if reward == 0.1 and not memory['key_picked_up']:
        memory['key_picked_up'] = True
    description = "Measures the directness of the path taken to the key."
    return {'result': memory['steps_to_key'] if memory['key_picked_up'] else 0, 'description': description}

def object_interaction_attempts(observation, action, reward, next_observation, memory):
    if 'interaction_attempts' not in memory:
        memory['interaction_attempts'] = 0
    if action in [3, 5]:  # Pick up or use key actions
        memory['interaction_attempts'] += 1
    description = "Counts how many times the agent attempts to interact with objects."
    return {'result': memory['interaction_attempts'], 'description': description}

def inventory_management_efficiency(observation, action, reward, next_observation, memory):
    if 'key_use_efficiency' not in memory:
        memory['key_use_efficiency'] = 0
    if action == 5 and 'key_picked_up' in memory and memory['key_picked_up']:  # Using key
        memory['key_use_efficiency'] += 1
    description = "Tracks how effectively the agent manages its inventory, particularly with the key."
    return {'result': memory['key_use_efficiency'], 'description': description}

def door_navigation_efficiency(observation, action, reward, next_observation, memory):
    if 'steps_after_key' not in memory:
        memory['steps_after_key'] = 0
        memory['door_reached'] = False
    if memory['key_picked_up'] and not memory['door_reached']:
        memory['steps_after_key'] += 1
    if action == 5 and reward == 0.2:  # Door opened
        memory['door_reached'] = True
    description = "Measures the efficiency of navigating to the door after picking up the key."
    return {'result': memory['steps_after_key'] if memory['door_reached'] else 0, 'description': description}

def tile_revisitation_rate(observation, action, reward, next_observation, memory):
    if 'visited_tiles' not in memory:
        memory['visited_tiles'] = set()
        memory['revisits'] = 0
    current_position = (observation['agent']['image'][3][6][0], observation['agent']['image'][3][6][1])
    if current_position in memory['visited_tiles']:
        memory['revisits'] += 1
    else:
        memory['visited_tiles'].add(current_position)
    total_visits = len(memory['visited_tiles']) + memory['revisits']
    revisitation_rate = memory['revisits'] / total_visits if total_visits else 0
    description = "Measures how frequently the agent revisits already explored tiles."
    return {'result': revisitation_rate, 'description': description}



metrics = {
    "steps_to_pickup_key": steps_to_key_pickup,
    "door_interaction_efficiency": door_interaction_efficiency,
    "exploration_efficiency": exploration_efficiency,
    "successful_box_pickup": successful_box_pickup,
    "path_efficiency_to_key": path_efficiency_to_key,
    "object_interaction_attempts": object_interaction_attempts,
    "inventory_management_efficiency": inventory_management_efficiency,
    "door_navigation_efficiency": door_navigation_efficiency,
    "tile_revisitation_rate": tile_revisitation_rate,
}