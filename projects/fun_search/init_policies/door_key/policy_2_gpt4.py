import numpy as np

class Policy:
    def __init__(self):
        self.has_key = False
        self.opened_door = False
        self.report = {'steps': 0, 'actions': [], 'rewards': 0, 'success': False, 'change_direction_due_to_stuck': 0, 'key_encounters': 0, 'door_encounters': 0}
        self.last_observation = None
        self.stuck_counter = 0  # Counter to detect if stuck in a loop
    
    def act(self, observation):
        direction = observation['direction']
        image = observation['image']
        forward_tile = image[3, 5, :]
        
        if forward_tile[0] == 5 and not self.has_key:
            self.report['key_encounters'] += 1
            return 3
        
        if self.has_key and forward_tile[0] == 4 and forward_tile[2] == 2:
            return 5
        
        if forward_tile[0] == 4 and forward_tile[2] == 0:
            return 2
        
        if forward_tile[0] == 7:
            return 3
        
        if forward_tile[0] == 1 or forward_tile[0] == 3:
            return 2
        
        # Detect if stuck and try to move differently
        if self.last_observation is not None and np.array_equal(observation['image'], self.last_observation['image']):
            self.stuck_counter += 1
            if self.stuck_counter > 3:  # Assume stuck after 3 repeated observations
                self.stuck_counter = 0
                self.report['change_direction_due_to_stuck'] += 1
                return np.random.choice([0, 1])  # Randomly choose to turn left or right
        
        self.stuck_counter = 0  # Reset counter if not stuck
        return 1  # Default action to turn right for exploration
    
    def update(self, observation, action, reward, next_observation):
        self.report['steps'] += 1
        self.report['actions'].append(action)
        self.report['rewards'] += reward
        self.last_observation = observation
        
        if action == 3 and next_observation['image'][3, 5, 0] == 5:
            self.has_key = True
            self.report['rewards'] += 0.1
        
        if action == 5 and next_observation['image'][3, 5, 2] == 0:
            self.opened_door = True
            self.report['rewards'] += 0.2
            self.report['door_encounters'] += 1
        
        if action == 3 and next_observation['image'][3, 5, 0] == 7:
            self.report['success'] = True
    
    def prepare_report(self) -> str:
        report_str = f"Total Steps: {self.report['steps']}\n"
        report_str += f"Actions Taken: {self.report['actions']}\n"
        report_str += f"Total Rewards: {self.report['rewards']}\n"
        report_str += f"Success: {self.report['success']}\n"
        report_str += f"Changes in Direction Due to Stuck: {self.report['change_direction_due_to_stuck']}\n"
        report_str += f"Key Encounters: {self.report['key_encounters']}\n"
        report_str += f"Door Encounters: {self.report['door_encounters']}"
        return report_str
