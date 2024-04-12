class WorldModel:
    def __init__(self):
        # Precondition descriptions for actions 3 (pickup) and 4 (drop)
        self.preconditions = {
            3: "Must be standing in tile adjacent to object and facing it, and cannot be holding another object",
            4: "Must be holding item. Tile you are facing must be empty"
        }

    def predict(self, observation, action):
        valid = False  # Initially, assume the action is not valid.
        new_observation = observation.copy()  # Make a copy of the observation to modify.
        
        # Handling pickup action.
        if action == 3:
            # Check if the precondition for pickup is met.
            if not observation['inv'] and observation['agent']['image'][3][5][0] not in (0, 1, 2, 3, 8, 9):
                new_observation['inv'].append(observation['agent']['image'][3][5][0])  # Pickup item.
                new_observation['agent']['image'][3][5] = (1, 0, 0)  # The tile becomes empty.
                valid = True
                
        # Handling drop action.
        elif action == 4:
            # Check if the precondition for drop is met.
            if observation['inv'] and observation['agent']['image'][3][5] == (1, 0, 0):  # Empty tile in front.
                item = new_observation['inv'].pop()  # Drop item.
                new_observation['agent']['image'][3][5] = (item, 0, 0)  # Place item on the tile.
                valid = True
                
        return {"observation": new_observation, "valid": valid}
