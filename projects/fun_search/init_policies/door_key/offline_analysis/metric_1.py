class Metrics:
    def metrics(self, trajectories):
        total_rewards = 0
        successes = 0
        total_steps = 0
        total_steps_success = 0
        keys_picked_up = 0
        doors_opened = 0
        total_exploration = 0

        for trajectory in trajectories:
            rewards = trajectory['rewards']
            actions = trajectory['actions']
            observations = trajectory['observations']

            total_rewards += sum(rewards)
            total_steps += len(actions)
            
            if sum(rewards) >= 1:  # Assuming success is indicated by reaching a total reward of 1 or more
                successes += 1
                total_steps_success += len(actions)

            # Track key pickups and door openings
            has_picked_key = any(reward == 0.1 for reward in rewards)
            if has_picked_key:
                keys_picked_up += 1

            has_opened_door = any(reward == 0.2 for reward in rewards)
            if has_opened_door:
                doors_opened += 1

            # Calculate exploration
            explored_tiles = set()
            for obs in observations:
                for row in obs['agent']['image']:
                    for tile in row:
                        explored_tiles.add(tuple(tile))
            total_exploration += len(explored_tiles)
        
        num_trajectories = len(trajectories)
        average_reward = total_rewards / num_trajectories
        success_rate = successes / num_trajectories
        average_steps_success = total_steps_success / successes if successes > 0 else 0
        average_keys_picked_up = keys_picked_up / num_trajectories
        average_doors_opened = doors_opened / num_trajectories
        average_exploration = total_exploration / num_trajectories

        return {
            'Average Reward': average_reward,
            'Success Rate': success_rate,
            'Average Steps to Success': average_steps_success,
            'Average Keys Picked Up': average_keys_picked_up,
            'Average Doors Opened': average_doors_opened,
            'Average Exploration': average_exploration
        }
