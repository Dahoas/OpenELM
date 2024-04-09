class Metrics:
    def metrics(self, trajectories):
        total_rewards = 0
        successes = 0
        total_steps = 0
        keys_picked_up = 0
        doors_opened = 0
        total_exploration = 0
        total_door_interactions = 0
        failed_door_interactions = 0
        failed_pickup_attempts = 0
        movement_after_key = 0

        for trajectory in trajectories:
            rewards = trajectory['rewards']
            actions = trajectory['actions']
            observations = trajectory['observations']
            key_possessed = False
            door_interaction_initiated = False
            
            total_rewards += sum(rewards)
            total_steps += len(actions)
            
            if sum(rewards) >= 1:  # Assuming success is indicated by reaching a total reward of 1 or more
                successes += 1

            explored_tiles = set()
            for i, obs in enumerate(observations[:-1]):
                image = obs['agent']['image']
                for row in image:
                    for tile in row:
                        explored_tiles.add(tuple(tile))
                        
                # Track key pickups
                if 5 in obs['inv'] and not key_possessed:
                    keys_picked_up += 1
                    key_possessed = True

                # Track door interactions
                if image[3][5][0] == 4:  # If facing a door
                    door_interaction_initiated = True
                    total_door_interactions += 1
                    if actions[i] != 5:  # If the action taken is not 'toggle key'
                        failed_door_interactions += 1
                
                # Track failed pickup attempts
                if actions[i] == 3 and image[3][5][0] not in [5, 7]:  # If attempting to pickup but no key/box is in front
                    failed_pickup_attempts += 1
                
                # Track movement after picking up key
                if key_possessed and actions[i] == 2:  # If moving forward after picking up key
                    movement_after_key += 1
            
            total_exploration += len(explored_tiles)
        
        num_trajectories = len(trajectories)
        average_reward = total_rewards / num_trajectories
        success_rate = successes / num_trajectories
        average_keys_picked_up = keys_picked_up / num_trajectories
        average_doors_opened = doors_opened / num_trajectories
        average_exploration = total_exploration / num_trajectories
        average_door_interactions = total_door_interactions / num_trajectories
        average_failed_door_interactions = failed_door_interactions / num_trajectories
        average_failed_pickup_attempts = failed_pickup_attempts / num_trajectories
        average_movement_after_key = movement_after_key / keys_picked_up if keys_picked_up > 0 else 0

        return {
            'Average Reward': average_reward,
            'Success Rate': success_rate,
            'Average Keys Picked Up': average_keys_picked_up,
            'Average Doors Opened': average_doors_opened,
            'Average Exploration': average_exploration,
            'Average Door Interactions': average_door_interactions,
            'Average Failed Door Interactions': average_failed_door_interactions,
            'Average Failed Pickup Attempts': average_failed_pickup_attempts,
            'Average Movement After Key': average_movement_after_key
        }
