import numpy as np

class Policy:
    def __init__(self):
        pass
    
    def act(self, observation):
        act = input("Action: ")
        return int(act)
    
    def update(self, observation, action, reward, next_observation):
        pass