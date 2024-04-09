import numpy as np

class Policy:
    def __init__(self):
        self.memory = {
            'key_picked_up': False,
            'door_opened': False,
            'in_front_of_box': False,
            'picked_up_box': False,
            'moved_to_goal': False,
            'step_count': 0,
            'steps_to_key': 0,
            'steps_to_door': 0,
            'steps_to_box_interaction': 0,
            'failed_pickup_attempts': 0,
            'unnecessary_actions': 0,
            'door_interactions_post_opening': 0,
            'inventory_changes': 0
        }
        self.metrics = {
            'steps_to_key': [],
            'steps_to_door': [],
            'steps_to_box_interaction': [],
            'failed_pickup_attempts': 0,
            'unnecessary_actions': 0,
            'door_interactions_post_opening': 0,
            'inventory_changes': 0
        }
        self.state = 'searching_for_key'  # Initial state

    def act(self, observation):
        self.memory['step_count'] += 1

        if observation['inv']:
            if observation['inv'][0] == 7 and not self.memory['picked_up_box']:
                self.memory['picked_up_box'] = True
                self.memory['inventory_changes'] += 1
                return 4  # Drop the box to achieve the goal
        
        if observation['agent']['image'][3][5][0] == 5 and not self.memory['key_picked_up']:
            self.memory['key_picked_up'] = True
            self.memory['inventory_changes'] += 1
            self.state = 'moving_to_door'
            self.metrics['steps_to_key'].append(self.memory['steps_to_key'])
            return 3  # Pick up key
        
        if self.memory['key_picked_up'] and observation['agent']['image'][3][5][0] == 4 and observation['agent']['image'][3][5][2] == 2:
            self.memory['door_opened'] = True
            self.state = 'searching_for_box'
            self.metrics['steps_to_door'].append(self.memory['steps_to_door'])
            return 5  # Open door
        
        if self.memory['in_front_of_box'] and not self.memory['picked_up_box']:
            self.memory['in_front_of_box'] = False
            self.memory['inventory_changes'] += 1
            return 3  # Pick up box
        
        if self.memory['door_opened'] and observation['agent']['image'][3][5][0] == 7:
            self.memory['in_front_of_box'] = True
            self.metrics['steps_to_box_interaction'].append(self.memory['steps_to_box_interaction'])
            return 3  # Attempt to pick up the box
        
        if self.memory['door_opened'] and not self.memory['moved_to_goal']:
            self.memory['moved_to_goal'] = True
            return 2  # Move forward towards the goal
        
        # Default action logic
        move = np.random.choice([0, 1, 2])
        if move == 2 and observation['agent']['image'][3][5][0] != 1:
            self.memory['unnecessary_actions'] += 1  # Increment if move forward is not possible
        return move

    def update(self, observation, action, reward, next_observation):
        # Update counters based on the current state
        if self.state == 'searching_for_key':
            self.memory['steps_to_key'] += 1
        elif self.state == 'moving_to_door':
            self.memory['steps_to_door'] += 1
        elif self.state == 'searching_for_box':
            self.memory['steps_to_box_interaction'] += 1
        # Track failed pickup attempts
        if action == 3 and next_observation['agent']['image'][3][5][0] not in [5, 7]:  # If action is pickup but the tile is not key or box
            self.memory['failed_pickup_attempts'] += 1

    @property
    def report(self):
        # Calculate additional metrics
        efficiency_ratio = self.memory['step_count'] / (sum(self.metrics['steps_to_key']) + sum(self.metrics['steps_to_door']) + sum(self.metrics['steps_to_box_interaction']) + 1)
        return {
            'key_picked_up/mean': int(self.memory['key_picked_up']),
            'door_opened/mean': int(self.memory['door_opened']),
            'box_picked_up/mean': int(self.memory['picked_up_box']),
            'moved_to_goal/mean': int(self.memory['moved_to_goal']),
            'steps_to_key/mean': np.mean(self.metrics['steps_to_key']) if self.metrics['steps_to_key'] else 0,
            'steps_to_door/mean': np.mean(self.metrics['steps_to_door']) if self.metrics['steps_to_door'] else 0,
            'steps_to_box_interaction/mean': np.mean(self.metrics['steps_to_box_interaction']) if self.metrics['steps_to_box_interaction'] else 0,
            'failed_pickup_attempts/mean': self.memory['failed_pickup_attempts'],
            'unnecessary_actions/mean': self.memory['unnecessary_actions'],
            'door_interactions_post_opening/mean': self.memory['door_interactions_post_opening'],
            'inventory_changes/mean': self.memory['inventory_changes'],
            'efficiency_ratio/mean': efficiency_ratio
        }
