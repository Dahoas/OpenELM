import numpy as np
from collections import defaultdict

class Policy:
    def __init__(self):
        self.notes = []
        self.state_memory = []
        self.previous_actions = defaultdict(int)  # Keep track of how often each action is taken

        # Priority list for actions based on current goals
        self.priority_actions = [
            'collect_wood', 'collect_stone', 'make_wood_pickaxe', 'collect_coal', 'make_stone_pickaxe',
            'collect_iron', 'make_iron_pickaxe', 'collect_diamond', 'make_wood_sword', 'make_stone_sword',
            'make_iron_sword', 'collect_drink', 'eat_plant', 'eat_cow', 'defeat_skeleton', 'defeat_zombie',
            'place_stone', 'place_table', 'place_furnace', 'place_plant'
        ]

        # Mapping of actions to integers
        self.action_map = {
            'noop': 0,
            'move_left': 1,
            'move_right': 2,
            'move_up': 3,
            'move_down': 4,
            'do': 5,
            'sleep': 6,
            'place_stone': 7,
            'place_table': 8,
            'place_furnace': 9,
            'place_plant': 10,
            'make_wood_pickaxe': 11,
            'make_stone_pickaxe': 12,
            'make_iron_pickaxe': 13,
            'make_wood_sword': 14,
            'make_stone_sword': 15,
            'make_iron_sword': 16
        }

    def act(self, observation):
        # Placeholder for decision-making process, should be replaced with logic to choose actions
        action = self.action_map['noop']  # Default action is no-operation
        self.notes.append("Acting based on current observation.")
        return action

    def update(self, observation, action, reward, next_observation):
        # Update internal state based on the outcome of the last action
        self.state_memory.append((observation, action, reward, next_observation))
        self.previous_actions[action] += 1
        if reward != 0:
            self.notes.append(f"Received reward: {reward} for action: {action}")
        else:
            self.notes.append(f"No reward received for action: {action}")

    def produce_report(self):
        # Generate a report based on the notes and actions taken
        action_summary = ", ".join([f"{action}: {count}" for action, count in self.previous_actions.items()])
        report = f"Action Summary: {action_summary}\nNotes Summary: {'; '.join(self.notes)}"
        return report
