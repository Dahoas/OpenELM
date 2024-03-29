class Policy:
    def __init__(self):
        self.notes = []
        self.state_memory = {}  # Stores observed states and actions taken
        self.achievement_memory = {}  # Tracks achievements and attempts to achieve them
        self.last_action = None
        self.last_observation = None
        self.health_threshold = 5  # Health level to start searching for food or water
    
    def act(self, observation):
        player = observation['player']
        action = 0  # default noop

        # Prioritize actions based on player needs
        if player.health <= self.health_threshold:
            self.notes.append("Low health, searching for resources")
            action = self.find_resources(player)
        elif player._hunger > 5 or player._thirst > 5:
            self.notes.append("Needs food or water")
            action = self.find_resources(player)
        elif player.sleeping or player._fatigue > 5:
            self.notes.append("Needs rest")
            action = 6  # sleep
        else:
            action = self.explore_or_gather(player)
        
        self.last_action = action
        return action
    
    def update(self, observation, action, reward, next_observation):
        # Update memories with outcomes of last action
        self.state_memory[str(observation)] = action
        if reward > 0:
            achievement = self.identify_achievement(observation, next_observation)
            if achievement:
                self.achievement_memory[achievement] = self.achievement_memory.get(achievement, 0) + 1
        self.last_observation = next_observation
    
    def find_resources(self, player):
        # Implement a simple strategy to look for resources based on player status
        if player._hunger > 5:
            return 5  # do action to search for food
        elif player._thirst > 5:
            return 5  # do action to search for water
        else:
            return self.explore_randomly()

    def explore_randomly(self):
        # Explore randomly if no immediate needs
        import random
        return random.choice([1, 2, 3, 4])

    def explore_or_gather(self, player):
        # Strategy to explore new areas or gather materials
        if any(val == 0 for val in player.inventory.values() if val is not None):
            return 5  # gather resources if any resource is at 0
        else:
            return self.explore_randomly()

    def identify_achievement(self, observation, next_observation):
        # Identify what achievement was possibly unlocked
        for achievement in observation['player'].achievements:
            if next_observation['player'].achievements[achievement] > observation['player'].achievements[achievement]:
                return achievement
        return None
    
    def produce_report(self):
        report_lines = [
            f"Total unique states encountered: {len(self.state_memory)}",
            f"Total achievements unlocked: {sum(self.achievement_memory.values())}",
            f"Details of achievements: {self.achievement_memory}",
            f"Total notes taken: {len(self.notes)}"
        ]
        return "\n".join(report_lines)
