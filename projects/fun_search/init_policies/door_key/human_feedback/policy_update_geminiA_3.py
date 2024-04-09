import numpy as np

class Policy:
    def __init__(self):
        self.holding_key = False
        self.key_dropped_in_front_of_door = False
        self.facing_door = False
        self.prev_action = None
        self.prev_observation = None
        self.prev_reward = 0
        self.explore_count = 0

    def act(self, observation):
        # Prioritize picking up the box, even if door is open
        if observation['image'][3][5][0] == 7: 
            return 3 # Pick up the box

        # Logic for dropping the key in a suitable location
        if self.holding_key:
            if observation['image'][3][5][0] == 4 and observation['image'][3][5][2] == 0:  # Open door
                return 2  # Move forward
            elif not self.key_dropped_in_front_of_door and observation['image'][3][5][0] == 1:  # Empty space
                self.key_dropped_in_front_of_door = True
                return 4  # Drop the key
            elif observation['image'][3][5][0] == 7:  # Box in front
                # Find an empty spot around and drop the key
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    if observation['image'][3 + dx][5 + dy][0] == 1:
                        if dx == 0:  # Need to turn before dropping
                            return int(dy > 0)  # 1 if right turn, 0 if left turn 
                        else:  
                            return 2  # Move forward
                return 4  # Drop if no space found (sub-optimal, but avoids infinite loops)

        # Picking up the key
        if not self.holding_key and observation['image'][3][5][0] == 5:
            return 3  # Pick up the key

        # Opening the door
        if self.holding_key and observation['image'][3][5][0] == 4 and observation['image'][3][5][2] == 2:
            return 5  # Use the key 

        # Exploration strategies
        if self.explore_count < 10:  # Explore for 10 steps initially
            self.explore_count += 1
            return np.random.choice([0, 1, 2])  

        # Avoid getting stuck 
        if self.prev_action == 2 and self.prev_reward < 0.1 and np.random.rand() < 0.4:
            return np.random.choice([0, 0, 1, 1]) 

        # Move towards the target, prioritizing the key 
        if not self.holding_key:
            target_object = 5  # Key
        elif not self.key_dropped_in_front_of_door:
            target_object = 1  # Empty space to drop key
        else:
            target_object = 4  # Door

        if observation['image'][3][5][0] == target_object:
            return 2  
        else:
            return move_towards((3, 5), observation['image'][3][5])  

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