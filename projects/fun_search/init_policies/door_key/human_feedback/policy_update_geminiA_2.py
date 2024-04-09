import numpy as np

class Policy:
    def __init__(self):
        self.holding_key = False
        self.facing_door = False
        self.prev_action = None
        self.prev_observation = None
        self.prev_reward = 0
        self.explore_count = 0

    def act(self, observation):
        # Prioritize picking up the box, even if door is open
        if observation['image'][3][5][0] == 7: 
            return 3  # Pick up the box

        # Other action decisions remain the same as before...
        if not self.holding_key and observation['image'][3][5][0] == 5:
            return 3  # Pick up the key
        if self.holding_key and observation['image'][3][5][0] == 4 and observation['image'][3][5][2] == 2:
            return 5  # Toggle key to open the door

        # Prioritize entering the open door if carrying the box
        if self.holding_key and observation['image'][3][5][0] == 4 and observation['image'][3][5][2] == 0:
            return 2  # Move forward if the door is open

        # Drop the key if standing in front of the box
        if self.holding_key and observation['image'][3][5][0] == 7:
            return 4  # Drop the key

        # Exploration strategies
        if self.explore_count < 10:  # Explore for 10 steps initially
            self.explore_count += 1
            return np.random.choice([0, 1, 2])  # Random turning

        # Stuck prevention – Backtrack a step if no progress towards key/door
        if self.prev_action == 2 and self.prev_reward < 0.1 and np.random.rand() < 0.5:  
            return np.random.choice([0, 0, 1, 1])  # Turn around

        # Move towards the target, prioritizing the key over the box
        if not self.holding_key:
            target_object = 5  # Key
        else:
            target_object = 4  # Door

        if observation['image'][3][5][0] == target_object:
            return 2  # Move forward if target is directly ahead
        else:
            return move_towards((3, 5), observation['image'][3][5])  # Move towards the target position

    def update(self, observation, action, reward, next_observation):
        self.prev_action = action
        self.prev_observation = observation
        self.prev_reward = reward
        if action == 3 and observation['image'][3][5][0] == 5:
            self.holding_key = True
        if action == 4:
            self.holding_key = False


def move_towards(target_position, next_tile):
    target_x, target_y = target_position
    if target_y < 6 and next_tile[0] == 1:
        return 2  # If target is in front of me and the next stil is empty, go forward
    else:
        if target_x < 3:
            return 0  # If target is to my left, turn left
        else:
            return 1  # If target is to my right, turn right