import numpy as np

class Policy:
    def __init__(self):
        self.has_key = False
        self.opened_door = False
        self.holding_item = False
        self.report = {'steps': 0, 'actions': [], 'rewards': 0, 'success': False, 'change_direction_due_to_stuck': 0, 'key_encounters': 0, 'door_encounters': 0, 'picked_up_key': False, 'opened_door': False}
        self.last_observation = None
        self.stuck_counter = 0  # Counter to detect if stuck in a loop
        self.direction_attempts = [0, 0, 0, 0]  # Keep track of attempted directions to avoid loops
    
    def act(self, observation):
        direction = observation['direction']
        image = observation['image']
        forward_tile = image[3, 5, :]
        
        # Handle key pickup
        if forward_tile[0] == 5 and not self.has_key:
            self.report['key_encounters'] += 1
            return 3  # Pickup key
        
        # Use key if facing a locked door
        if self.has_key and forward_tile[0] == 4 and forward_tile[2] == 2:
            self.report['door_encounters'] += 1
            return 5  # Unlock door
        
        # Move forward if the path is clear (floor or opened door)
        if forward_tile[0] in [1, 3] or (forward_tile[0] == 4 and forward_tile[2] == 0):
            self.direction_attempts = [0, 0, 0, 0]  # Reset direction attempts after moving forward
            return 2  # Move forward
        
        # Pick up the box if it's in front
        if forward_tile[0] == 7 and not self.holding_item:
            return 3  # Pickup box
        
        # Change direction if facing a wall or closed door
        if forward_tile[0] in [2, 4]:
            return self.choose_direction_change(direction)
        
        # Default action if none of the above conditions are met
        return self.choose_direction_change(direction)
    
    def choose_direction_change(self, current_direction):
        # Attempt to turn in a new direction that hasn't been tried recently
        for i in range(4):
            if self.direction_attempts[(current_direction + i) % 4] == 0:
                self.direction_attempts[(current_direction + i) % 4] = 1
                return (i % 2) + 1  # Returns 1 for right turn, 2 for left turn based on iteration
        self.direction_attempts = [0, 0, 0, 0]  # Reset if all directions have been attempted
        return np.random.choice([0, 1])  # Randomly choose to turn left or right if stuck
        
    def update(self, observation, action, reward, next_observation):
        self.report['steps'] += 1
        self.report['actions'].append(action)
        self.report['rewards'] += reward
        self.last_observation = observation
        
        # Update flags based on actions
        if action == 3 and next_observation['image'][3, 5, 0] == 5:
            self.has_key = True
            self.report['picked_up_key'] = True
            self.report['rewards'] += 0.1
        
        if action == 5:
            self.opened_door = True
            self.report['opened_door'] = True
            self.report['rewards'] += 0.2
        
        if action == 3 and next_observation['image'][3, 5, 0] == 7:
            self.holding_item = True
            self.report['success'] = True
    
    def prepare_report(self) -> str:
        report_str = f"Total Steps: {self.report['steps']}\n"
        report_str += f"Actions Taken: {self.report['actions']}\n"
        report_str += f"Total Rewards: {self.report['rewards']}\n"
        report_str += f"Success: {self.report['success']}\n"
        report_str += f"Changes in Direction Due to Stuck: {self.report['change_direction_due_to_stuck']}\n"
        report_str += f"Key Encounters: {self.report['key_encounters']}\n"
        report_str += f"Door Encounters: {self.report['door_encounters']}\n"
        report_str += f"Picked Up Key: {self.report['picked_up_key']}\n"
        report_str += f"Opened Door: {self.report['opened_door']}"
        return report_str
