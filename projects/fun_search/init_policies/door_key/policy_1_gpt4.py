import numpy as np

class Policy:
    def __init__(self):
        # Initialize any persistent state or memory here
        self.has_key = False
        self.opened_door = False
        self.report = {'steps': 0, 'actions': [], 'rewards': 0, 'success': False}
    
    def act(self, observation):
        # Extract the relevant information from the observation
        direction = observation['direction']
        image = observation['image']
        forward_tile = image[3, 5, :]
        
        # If we see a key in front of us, pick it up
        if forward_tile[0] == 5 and not self.has_key:
            return 3  # Action to pick up the key
        
        # If we have the key and see a closed door in front of us, try to open it
        if self.has_key and forward_tile[0] == 4 and forward_tile[2] == 2:
            return 5  # Action to use the key
        
        # If the door is open, move forward into the next room
        if forward_tile[0] == 4 and forward_tile[2] == 0:
            return 2  # Action to move forward
        
        # If we see the box in front of us, pick it up
        if forward_tile[0] == 7:
            return 3  # Action to pick up the box
        
        # If the forward tile is empty, move forward
        if forward_tile[0] == 1 or forward_tile[0] == 3:
            return 2  # Action to move forward
        
        # Otherwise, turn right to explore
        return 1  # Action to turn right
    
    def update(self, observation, action, reward, next_observation):
        # Update the policy's state based on the action taken and the results
        self.report['steps'] += 1
        self.report['actions'].append(action)
        self.report['rewards'] += reward
        
        if action == 3 and next_observation['image'][3, 5, 0] == 5:
            self.has_key = True
            self.report['rewards'] += 0.1  # Additional reward for picking up the key
        
        if action == 5 and next_observation['image'][3, 5, 2] == 0:
            self.opened_door = True
            self.report['rewards'] += 0.2  # Additional reward for opening the door
        
        # Update success status if the box is picked up
        if action == 3 and next_observation['image'][3, 5, 0] == 7:
            self.report['success'] = True
    
    def prepare_report(self) -> str:
        # Format the report to be readable
        report_str = f"Total Steps: {self.report['steps']}\n"
        report_str += f"Actions Taken: {self.report['actions']}\n"
        report_str += f"Total Rewards: {self.report['rewards']}\n"
        report_str += f"Success: {self.report['success']}\n"
        return report_str
