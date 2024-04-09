class Metrics:
    def metrics(self, trajectories):
        success_count = 0
        total_rewards = 0
        door_interactions = 0
        key_pickups = 0
        useless_actions = 0
        box_interactions = 0
        direction_changes = 0
        steps_post_door = 0
        distances_to_box_post_door = []

        for trajectory in trajectories:
            rewards = trajectory['rewards']
            actions = trajectory['actions']
            observations = trajectory['observations']

            total_rewards += sum(rewards)
            if rewards[-1] > 0:  # Assuming positive reward only on success
                success_count += 1

            last_direction = observations[0]['agent']['direction']
            door_opened = False
            steps_since_door_opened = 0
            min_distance_to_box_after_door = float('inf')

            for i, (action, reward) in enumerate(zip(actions, rewards)):
                current_obs = observations[i]
                next_obs = observations[i + 1]

                # Check for door interaction
                if current_obs['agent']['image'][3][5][0] == 4 and action == 5:
                    door_interactions += 1
                    door_opened = True  # Assuming the action succeeded

                # Check for key pickups
                if current_obs['agent']['image'][3][5][0] == 5 and action == 3:
                    key_pickups += 1

                # Check for box interactions
                if current_obs['agent']['image'][3][5][0] == 7 and action == 3:
                    box_interactions += 1

                # Check for direction changes
                if action == 0 or action == 1:
                    if last_direction != next_obs['agent']['direction']:
                        direction_changes += 1
                        last_direction = next_obs['agent']['direction']

                # Check for useless actions
                if reward == 0 and (current_obs == next_obs or action in [0, 1, 4]):
                    useless_actions += 1

                # Track steps and distance to box after door is opened
                if door_opened:
                    steps_since_door_opened += 1
                    # Assuming we can calculate distance to the box (this part is pseudo-code)
                    # distance_to_box = calculate_distance_to_box(next_obs)
                    # min_distance_to_box_after_door = min(min_distance_to_box_after_door, distance_to_box)

            if door_opened:
                steps_post_door += steps_since_door_opened
                # distances_to_box_post_door.append(min_distance_to_box_after_door)

        success_rate = success_count / len(trajectories) if trajectories else 0
        average_reward = total_rewards / len(trajectories) if trajectories else 0
        # average_distance_to_box_post_door = sum(distances_to_box_post_door) / len(distances_to_box_post_door) if distances_to_box_post_door else 0

        return {
            'Success Rate': success_rate,
            'Average Reward': average_reward,
            'Door Interactions': door_interactions,
            'Key Pickups': key_pickups,
            'Useless Actions': useless_actions,
            'Box Interactions': box_interactions,
            'Direction Changes': direction_changes,
            'Average Steps After Door Open': steps_post_door / door_interactions if door_interactions else 0,
            # 'Average Distance to Box Post-Door': average_distance_to_box_post_door
        }
