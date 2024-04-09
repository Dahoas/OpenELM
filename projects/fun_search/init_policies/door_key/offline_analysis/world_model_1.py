class WorldModel:
    def __init__(self):
        pass

    def predict(self, observation, action):
        # Copy the observation to avoid modifying the original input
        new_observation = {'agent': observation['agent'].copy(), 'inv': observation['inv'].copy()}
        new_observation['agent']['image'] = observation['agent']['image'].copy()

        # Action 3: Pickup item
        if action == 3:
            # Preconditions: Agent must be facing an object that can be picked up and inventory must be empty
            if len(new_observation['inv']) == 0:  # Inventory is empty
                facing_tile = new_observation['agent']['image'][3][5]  # Tile agent is facing
                # Check if the facing tile has an item that can be picked up (not a wall, floor, or unseen)
                if facing_tile[0] in [5, 6, 7] and facing_tile[2] == 0:  # Item is present and is not a door
                    # Add item to inventory
                    new_observation['inv'].append(facing_tile[0])
                    # Remove the item from the world
                    new_observation['agent']['image'][3][5] = (1, facing_tile[1], 0)  # Replace with an empty tile of same color

        # Action 4: Drop item
        elif action == 4:
            # Preconditions: Must be holding an item and the tile facing must be empty
            if len(new_observation['inv']) > 0:  # Holding an item
                facing_tile = new_observation['agent']['image'][3][5]  # Tile agent is facing
                # Check if the facing tile is empty
                if facing_tile[0] == 1:  # Empty tile
                    # Remove item from inventory and place it in the world
                    item_to_drop = new_observation['inv'].pop()
                    new_observation['agent']['image'][3][5] = (item_to_drop, facing_tile[1], 0)  # Drop item with original color

        return new_observation