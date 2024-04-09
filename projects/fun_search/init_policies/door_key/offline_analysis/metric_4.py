class Metrics:
    def metrics(self, trajectories):
        total_trajectories = len(trajectories)
        success_count = 0
        total_rewards = 0
        door_interactions = 0
        key_pickups = 0
        useless_actions = 0
        failed_door_opens = 0
        key_holding_durations = []
        steps_after_door_open = []

        for trajectory in trajectories:
            rewards = trajectory['rewards']
            actions = trajectory['actions']
            observations = trajectory['observations']

            total_rewards += sum(rewards)
            success = rewards[-1] > 0  # Assuming success is indicated by the final reward
            if success:
                success_count += 1

            key_held = False
            key_used = False
            key_pickup_step = None
            door_opened_step = None
            for i, (action, reward) in enumerate(zip(actions, rewards)):
                obs = observations[i]
                forward_tile = obs['agent']['image'][3][5]
                inv = obs['inv']

                # Track key pickups
                if forward_tile[0] == 5 and action == 3:  # Key pickup action
                    key_pickups += 1
                    key_held = True
                    key_pickup_step = i

                # Track door interactions
                if forward_tile[0] == 4 and action == 5:  # Door interaction
                    if 5 in inv:  # Has key
                        door_interactions += 1
                        if door_opened_step is None:
                            door_opened_step = i
                    else:
                        failed_door_opens += 1

                if action == 4 and key_held:  # Key drop
                    key_held = False
                    if key_pickup_step is not None:
                        key_holding_durations.append(i - key_pickup_step)
                        key_pickup_step = None

                if reward == 0 and action in [0, 1, 2, 4]:  # Considered as a useless action
                    useless_actions += 1

            if door_opened_step is not None:
                steps_after_door_open.append(len(actions) - door_opened_step)

        return {
            'Success Rate': success_count / total_trajectories if total_trajectories else 0,
            'Average Reward': total_rewards / total_trajectories if total_trajectories else 0,
            'Door Interactions': door_interactions,
            'Key Pickups': key_pickups,
            'Useless Actions': useless_actions,
            'Failed Door Opens': failed_door_opens,
            'Average Key Holding Duration': sum(key_holding_durations) / len(key_holding_durations) if key_holding_durations else 0,
            'Average Steps After Door Open': sum(steps_after_door_open) / len(steps_after_door_open) if steps_after_door_open else 0
        }
