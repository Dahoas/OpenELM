import numpy as np

class Policy:
    def __init__(self):
        self.holding_key = False
        self.door_opened = False
        self.prev_action = None
        self.prev_observation = None
        self.prev_reward = 0
        self.explore_count = 0
        self.action_sequence = []
        self.sequence_index = 0
        self.trying_to_drop_key = False

    def act(self, observation):
        # Handle dropping the key if standing in front of the box with the key
        if self.holding_key and observation['image'][3][5][0] == 7:  # Box in front
            self.trying_to_drop_key = True
            # Turn around to find an empty spot to drop the key
            return 0  # Turn left to start looking for a spot

        if self.trying_to_drop_key:
            # If facing an empty tile after turning, drop the key
            if observation['image'][3][5][0] == 1:  # Empty space
                self.trying_to_drop_key = False
                return 4  # Drop the key
            else:
                # Keep turning to find an empty spot
                return 0  # Turn left
            
        # Prioritize picking up the box if it's directly in front and no key to drop
        if observation['image'][3][5][0] == 7 and not self.holding_key:
            return 3  # Pick up the box

        # Picking up the key
        if not self.holding_key and observation['image'][3][5][0] == 5:
            self.holding_key = True
            return 3  # Pick up the key

        # Opening the door
        if self.holding_key and observation['image'][3][5][0] == 4 and observation['image'][3][5][2] == 2:
            self.door_opened = True
            return 5  # Use the key to open the door

        # Move forward if the door is open
        if self.door_opened and observation['image'][3][5][0] in [1, 7]:  # Empty space or box
            return 2  # Move forward

        # Exploration strategies
        if self.explore_count < 20:  # Explore for 20 steps initially or if stuck
            self.explore_count += 1
            return np.random.choice([0, 1, 2])  # Turn left, turn right, or move forward randomly

        # If stuck or at a loss of what to do, turn randomly
        if self.prev_action == 2 and self.prev_reward < 0.1:
            return np.random.choice([0, 1])  # Turn left or right randomly

        # Default action to prevent getting stuck
        return np.random.choice([0, 1, 2])

    def update(self, observation, action, reward, next_observation):
        self.prev_action = action
        self.prev_observation = observation
        self.prev_reward = reward
        if action == 3 and observation['image'][3][5][0] == 5:
            self.holding_key = True
        if action == 4:
            self.holding_key = False
        if reward > 0:
            self.door_opened = True  # Assume door opened if reward is received for opening door
