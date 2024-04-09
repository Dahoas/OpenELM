import numpy as np

class Policy:
    def __init__(self):
        self.memory = {
            'key_picked_up': False,
            'door_opened': False,
            'door_opening': False,
            'door_unlocked': False,
            'box_picked_up': False
        }

        self.steps = 0
        self.actions = [0, 1, 2, 3, 4, 5]
        self.explore_count = 0
        self.explored_tiles = set()

    def act(self, observation):
        direction = observation['agent']['direction']
        image = observation['agent']['image']
        inv = observation['inv']

        current_tile = image[3][6]
        forward_tile = image[3][5]

        if forward_tile[0] == 5 and 3 not in inv:
            return 3  # Pickup key

        if forward_tile[0] == 4 and forward_tile[2] == 2 and 5 in inv and not self.memory['door_opening']:
            return 5  # Toggle key to open door

        if forward_tile[0] == 7 and 0 in inv:
            return 3  # Pickup box if available

        if forward_tile[0] not in [2, 9] and tuple(forward_tile) not in self.explored_tiles:
            self.explored_tiles.add(tuple(forward_tile))
            return 2  # Move forward if facing an unexplored tile

        # If holding key, door is open, and facing the box, drop the key before picking up the box
        if 5 in inv and self.memory['door_opened'] and forward_tile[0] == 7:
            return 4  # Drop the key

        # Exploration strategy
        self.explore_count += 1

        if self.explore_count % 3 == 0:
            return np.random.choice([0, 1])  # Randomly turn left or right every 3rd step

        if self.explore_count % 2 == 0:
            return 2  # Move forward every 2nd step

        return 2  # Move forward by default

    def update(self, observation, action, reward, next_observation):
        key_picked_up = self.memory['key_picked_up']
        door_opened = self.memory['door_opened']
        door_unlocked = self.memory['door_unlocked']
        door_opening = self.memory['door_opening']
        box_picked_up = self.memory['box_picked_up']

        self.steps += 1

        if not key_picked_up and reward == 0.1:
            self.memory['key_picked_up'] = True

        if not door_unlocked and reward == 0.3:
            self.memory['door_unlocked'] = True

        if not door_opening and reward == 0.2:
            self.memory['door_opening'] = True

        if not door_opened and reward == 0.1:
            self.memory['door_opened'] = True

        if not box_picked_up and reward == 0.9:
            self.memory['box_picked_up'] = True