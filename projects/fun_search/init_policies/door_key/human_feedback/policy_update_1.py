import numpy as np


class Policy:
    def __init__(self):
        self.holding_item = False
        self.has_key = False
        self.direction = 0  # 0: right, 1: down, 2: left, 3: up
        self.observed_objects = set()  # Keep track of observed objects
        self.last_action = None  # To track the last action, to avoid getting stuck

    def act(self, observation):
        if not self.holding_item:
            if observation["image"][3][5][0] == 5 and not self.has_key:
                # Pick up the key
                return 3
            elif self.has_key and observation["image"][3][5][0] == 4:
                # Use the key to open the door
                return 5
            elif 6 in self.observed_objects and observation["image"][3][5][0] == 7:
                # Pick up the box
                return 3
            elif 7 in self.observed_objects and observation["image"][3][5][0] == 8:
                # Drop the item
                return 4
            else:
                # Explore but avoid getting stuck
                if self.last_action in [0, 1]:
                    return 2  # Move forward if last action was turn
                return np.random.choice([0, 1, 2])  # Randomly turn or move forward
        else:
            if observation["image"][3][5][0] == 8:
                # Drop the item
                return 4
            else:
                return np.random.choice([0, 1, 2])  # Randomly turn or move forward

    def update(self, observation, action, reward, next_observation):
        self.last_action = action
        if action == 3:  # Picking up the key
            if self.holding_item:
                return
            elif next_observation["image"][3][5][0] == 5:
                self.holding_item = True
                self.observed_objects.add(5)
                return

        if action == 5:  # Using the key to open the door
            if self.has_key:
                if next_observation["image"][3][5][0] == 4:
                    next_observation["image"][3][5][2] = 0  # Set the door to be open
                    return

        if reward == 1:  # Successful completion
            self.holding_item = False
            self.has_key = False
            self.observed_objects.clear()
        elif action == 2:  # Moving forward
            if next_observation["image"][3][5][0] == 2:  # If the next cell is a wall
                self.direction = (self.direction + 1) % 4  # Turn right

        if reward == 0.1:  # Reward for picking up the key for the first time
            self.has_key = True
