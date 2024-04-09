import numpy as np

class Policy:
    def __init__(self):
        self.holding_key = False
        self.opened_door = False  # Track if the door has been opened
        self.holding_item = False
        self.prev_action = None
        self.prev_observation = None
        self.prev_reward = 0
        self.explore_count = 0
        self.steps_after_opening_door = 0  # Track steps after opening the door

    def act(self, observation):
        # Pick up the key if not holding it and it's in front
        if not self.holding_key and observation['image'][3][5][0] == 5:
            return 3
        # Open the door if holding the key and facing a locked door
        if self.holding_key and observation['image'][3][5][0] == 4 and observation['image'][3][5][2] == 2:
            return 5
        # After opening the door, ensure movement through it
        if self.opened_door and self.steps_after_opening_door < 3:
            self.steps_after_opening_door += 1
            return 2  # Move forward to go through the door
        # Pick up the box if not holding an item and it's in front
        if not self.holding_item and observation['image'][3][5][0] == 7:
            return 3
        # Drop the item if holding it and facing an empty tile
        if self.holding_item and observation['image'][3][5][0] == 1:
            return 4

        # Explore if not clear what to do, but focus on moving forward after opening the door
        if self.explore_count < 5:
            self.explore_count += 1
            return np.random.choice([0, 1, 2])
        else:
            # If facing a wall or obstacle, turn to find a way around
            if observation['image'][3][5][0] in [0, 2]:  # If unseen or wall, turn
                return np.random.choice([0, 1])
            else:
                return 2  # Move forward if the path is clear

    def update(self, observation, action, reward, next_observation):
        self.prev_action = action
        self.prev_observation = observation
        self.prev_reward = reward
        if action == 3 and observation['image'][3][5][0] == 5:
            self.holding_key = True
        if action == 5 and next_observation['image'][3][5][2] == 0:
            self.holding_key = False  # Update to reflect the key is used, not dropped
            self.opened_door = True  # Door is now open
        if action == 3 and observation['image'][3][5][0] == 7:
            self.holding_item = True
        if action == 4:
            self.holding_item = False
        if self.opened_door:
            self.steps_after_opening_door += 1  # Increment steps after opening the door
