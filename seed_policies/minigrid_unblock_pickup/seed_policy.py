import numpy as np

class Policy:
    def __init__(self):
        self.has_key = False
        self.has_moved_ball = False
        self.has_opened_door = False
        self.target_item = 6  # Start with the ball as a target

        self.is_moving_ball_routine = 0

    def act(self, observation):
        direction = observation['direction']
        image = observation['image']

        # Find target position and item in front of the agent
        target_position = np.where(image[:, :, 0] == self.target_item)
        next_tile = image[3, 5]
        item_in_front = next_tile[0]

        if self.is_moving_ball_routine:
            # Just picked up ball, need to try to turn around and drop it 
            if self.is_moving_ball_routine == 1:
                return 1
            # Drop ball if rotated from original tile and forward tile is empty
            elif self.is_moving_ball_routine > 1 and item_in_front == 1:
                return 4
            # Rotate again
            else:
                return 1
        
        # If the agent is not holding what it needs, turn or move to get it
        if not self.has_key or not self.has_moved_ball or not self.has_opened_door:
            if len(target_position[0]) > 0:  # If we see our target
                # If the target is directly in front, attempt to interact or move towards it
                if item_in_front == self.target_item:
                    if self.target_item == 4:
                        return 5  # Use key on door
                    else:
                        return 3  # Pick up the item or interact with it
                else:
                    return self.move_towards(target_position, next_tile)  # Move towards the target
            else:
                # If we don't see our target, turn right to search for it
                return 1  # Turn right

        # If the agent has the key, has moved the ball, and opened the door, then it's time to get the box
        else:
            box_position = np.where(image[:, :, 0] == 7)
            if len(box_position[0]) > 0:  # If we see the box
                if item_in_front == 7:  # If the box is directly in front
                    return 3  # Pick up the box
                else:
                    return self.move_towards(direction, box_position)  # Move towards the box
            else:
                # If we don't see the box, turn right to search for it
                return 1  # Turn right

    def update(self, observation, action, reward, next_observation):
        if self.is_moving_ball_routine:
            self.is_moving_ball_routine += 1

        if action == 3 and observation['image'][3, 5][0] == 6:
            self.is_moving_ball_routine += 1

        # Check if the last action was to drop the ball
        if action == 4 and self.is_moving_ball_routine:
            self.is_moving_ball_routine = 0
            self.target_item = 5
            
        # Check if the last action was to pick up the key
        if action == 3 and observation['image'][3, 5][0] == 5:
            self.has_key = True
            self.target_item = 4

        # Check if the door has been opened
        if self.has_key and not self.has_opened_door:
            door_position = np.where((observation['image'][:, :, 0] == 4) & (observation['image'][:, :, 2] < 2))
            if len(door_position[0]) > 0:  # If we see an open or unlocked door
                self.has_opened_door = True
                self.target_item = 7

    def move_towards(self, target_position, next_tile):
        target_x, target_y = target_position
        if target_y < 6 and next_tile[0] == 1:
            return 2  # If target is in front of me and the next stil is empty, go forward
        else:
            if target_x < 3:
                return 0  # If target is to my left, turn left
            else:
                return 1  # If target is to my right, turn right