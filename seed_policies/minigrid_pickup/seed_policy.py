import numpy as np

class Policy:
    def __init__(self):
        self.holding_key = False
        self.facing_door = False
        self.holding_item = False
        self.prev_action = None
        self.prev_observation = None
        self.prev_reward = 0
        self.explore_count = 0

    def act(self, observation):
        if not self.holding_key and observation['image'][3][5][0] == 5:
            return 3  # pick up the key
        if not self.holding_item and observation['image'][3][5][0] == 7:
            return 3  # pick up the box
        if self.holding_key and observation['image'][3][5][0] == 4 and observation['image'][3][5][2] == 2:
            return 5  # toggle key to open the door
        if self.holding_item and observation['image'][3][5][0] == 8:
            return 4  # drop item
        if not self.holding_key and observation['image'][3][5][0] == 9:
            if self.explore_count < 5:  # Explore by turning randomly for the first 5 steps
                self.explore_count += 1
                return np.random.choice([0, 1, 2])
            else:
                self.explore_count += 1
                return move_towards((3, 5), observation['image'][3][5])  # Move towards the target position
        if self.prev_action == 2 and self.prev_reward < 0.1:
            # If moving forward results in a low reward, explore by turning randomly
            return np.random.choice([0, 1, 2])
        return move_towards((3, 5), observation['image'][3][5])  # Move towards the target position

    def update(self, observation, action, reward, next_observation):
        self.prev_action = action
        self.prev_observation = observation
        self.prev_reward = reward
        if action == 3 and observation['image'][3][5][0] == 5:
            self.holding_key = True
        if action == 5 and next_observation['image'][3][5][2] == 0:
            self.holding_key = False
        if action == 3 and observation['image'][3][5][0] == 7:
            self.holding_item = True
        if action == 4:
            self.holding_item = False
