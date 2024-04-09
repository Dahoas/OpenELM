class Metrics:
    def metrics(self, trajectories):
        success_count = 0
        total_rewards = 0
        steps_in_success = []
        door_interactions = 0
        key_pickups = 0
        useless_actions = 0

        for trajectory in trajectories:
            rewards = trajectory['rewards']
            actions = trajectory['actions']
            observations = trajectory['observations']

            total_rewards += sum(rewards)
            if rewards[-1] > 0:  # Assuming positive reward only on success
                success_count += 1
                steps_in_success.append(len(actions))

            for i, (action, reward) in enumerate(zip(actions, rewards)):
                current_obs = observations[i]
                next_obs = observations[i + 1]

                # Check for door interaction
                if current_obs['agent']['image'][3][5][0] == 4 and action == 5:
                    door_interactions += 1

                # Check for key pickup
                if current_obs['agent']['image'][3][5][0] == 5 and action == 3:
                    key_pickups += 1

                # Check for useless actions
                if reward == 0 and (current_obs == next_obs or action in [0, 1, 4]):
                    useless_actions += 1

        success_rate = success_count / len(trajectories) if trajectories else 0
        average_reward = total_rewards / len(trajectories) if trajectories else 0
        average_steps_success = sum(steps_in_success) / len(steps_in_success) if steps_in_success else 0

        return {
            'Success Rate': success_rate,
            'Average Reward': average_reward,
            'Average Steps to Success': average_steps_success,
            'Door Interactions': door_interactions,
            'Key Pickups': key_pickups,
            'Useless Actions': useless_actions
        }
