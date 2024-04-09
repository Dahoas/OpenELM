import numpy as np

class Policy:
    def __init__(self):
        self.has_key = False
        self.opened_door = False
        self.target_reached = False
        self.prev_action = -1
        self.stuck_counter = 0
        self.report = {"steps": 0, "success": False, "actions_taken": [], "key_picked": False, "door_opened": False}
        self.turns_since_last_obstacle = 0
        self.forward_attempts_since_last_success = 0
        self.random_turn_flag = False

    def act(self, observation):
        direction = observation['direction']
        image = observation['image']
        forward_tile = image[3][5]
        
        # If the agent has been turning too much or attempted to move forward without success,
        # introduce a random action to escape potential loops
        if self.turns_since_last_obstacle >= 3 or self.forward_attempts_since_last_success >= 3:
            if not self.random_turn_flag:  # Only introduce a random action once per identified loop
                action = np.random.choice([0, 1, 2])
                self.report["actions_taken"].append("Escaping loop")
                self.random_turn_flag = True  # Prevent consecutive random actions
                return action

        if forward_tile[0] == 5:  # If key is in front
            action = 3  # Pickup key
            self.has_key = True
            self.report["key_picked"] = True
            self.report["actions_taken"].append("Picked up key")
        
        elif forward_tile[0] == 4 and forward_tile[2] == 2 and self.has_key:  # If locked door is in front and we have a key
            action = 5  # Open/Toggle door
            self.opened_door = True
            self.report["door_opened"] = True
            self.report["actions_taken"].append("Opened door")
        
        elif forward_tile[0] in [3, 7] or (forward_tile[0] == 4 and forward_tile[2] == 0):  # Move forward if the tile in front is empty, door is open, or the box is in front
            action = 2  # Move forward
            self.forward_attempts_since_last_success = 0  # Reset forward attempt counter on a successful forward move
            self.random_turn_flag = False  # Reset random action flag

            if forward_tile[0] == 7:  # If box is in front
                self.report["actions_taken"].append("Reached box")
                self.target_reached = True
                self.report["success"] = True
            else:
                self.report["actions_taken"].append("Moved forward")
        
        else:  # If facing an obstacle or a wall
            self.forward_attempts_since_last_success += 1  # Increment forward attempt counter
            if self.prev_action in [0, 1]:  # Prioritize moving forward if the last action was a turn
                action = 2
                self.report["actions_taken"].append("Attempt forward after turn")
            else:  # Randomly choose to turn left or right if not just turned
                action = np.random.choice([0, 1])
                self.report["actions_taken"].append("Random turn")
            self.turns_since_last_obstacle += 1  # Increment turns counter
        
        # Prepare for next iteration
        self.prev_action = action
        self.report["steps"] += 1
        
        return action

    def update(self, observation, action, reward, next_observation):
        # Reset turns counter if the action was not a turn
        if action != self.prev_action or action not in [0, 1]:
            self.turns_since_last_obstacle = 0
        
        # Special handling for door opening and box pickup as these actions directly lead towards goal
        if action == 3 and next_observation['image'][3][5][0] == 7:
            self.target_reached = True
            self.report["success"] = True
        elif action == 5:
            self.opened_door = True
            self.report["door_opened"] = True

    @staticmethod
    def prepare_report(reports: list[dict]) -> str:
        total_steps, successes, keys_picked, doors_opened = 0, 0, 0, 0
        for report in reports:
            total_steps += report.get("steps", 0)
            if report.get("success", False):
                successes += 1
            if report.get("key_picked", False):
                keys_picked += 1
            if report.get("door_opened", False):
                doors_opened += 1
        
        success_rate = successes / len(reports)
        avg_steps = total_steps / len(reports) if len(reports) > 0 else 0
        key_pick_rate = keys_picked / len(reports)
        door_open_rate = doors_opened / len(reports)
        
        summary = (f"Success Rate: {success_rate:.2%}, Avg. Steps: {avg_steps:.2f}, "
                   f"Key Pickup Rate: {key_pick_rate:.2%}, Door Open Rate: {door_open_rate:.2%}")
        return summary
