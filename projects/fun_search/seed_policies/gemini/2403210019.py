from collections import deque

class Policy:
  def __init__(self):
    self.state = None  # Current state of the environment
    self.prev_action = None  # Previously taken action
    self.resource_goals = {'wood': 3, 'stone': 3, 'coal': 1}  # Minimum resources needed
    self.resource_locations = {}  # Stores discovered resource locations (if any)
    self.need_rest = False  # Flag to indicate need for sleep
    self.need_food = False  # Flag to indicate need for food
    self.need_drink = False  # Flag to indicate need for drink
    self.action_history = deque(maxlen=10)  # Stores last 10 actions for loop detection
    self.resource_gathering_attempts = 0  # Tracks resource gathering attempts

  def act(self, observation):
    self.update_state(observation)
    self.check_needs()

    # Prioritize basic needs (food, drink, sleep)
    if self.need_food and self.player.inventory['food'] > 0:
      return 5  # Eat
    elif self.need_drink and self.player.inventory['drink'] > 0:
      return 5  # Drink
    elif self.need_rest and self.player.sleeping == False:
      return 6  # Sleep

    # Craft tools if resources allow
    if self.can_craft('wood_pickaxe'):
      return 11
    elif self.can_craft('stone_pickaxe'):
      return 12
    elif self.can_craft('iron_pickaxe'):
      return 13

    # Gather resources if needed
    if not self.has_resources():
      return self.gather_resources()

    # Explore or build based on remaining time
    if observation.env.step_count > 0.75 * 1e6:
      return self.explore()
    else:
      return self.build_shelter()

  def update(self, observation, action, reward, next_observation):
    self.prev_action = action
    self.action_history.append(action)

  @property
  def player(self):
    return self.state.player

  def update_state(self, observation):
    self.state = observation

  def check_needs(self):
    self.need_rest = self.player._fatigue > 5
    self.need_food = self.player._hunger > 5
    self.need_drink = self.player._thirst > 5

  def can_craft(self, item):
    recipe = self.get_crafting_recipe(item)
    return all(self.player.inventory[resource] >= amount for resource, amount in recipe.items())

  def get_crafting_recipe(self, item):
    recipes = {
        'wood_pickaxe': {'wood': 3},
        'stone_pickaxe': {'wood': 3, 'stone': 2},
        'iron_pickaxe': {'wood': 3, 'stone': 2, 'iron': 1}
    }
    return recipes.get(item, {})

  def has_resources(self):
    return all(self.player.inventory[resource] >= amount for resource, amount in self.resource_goals.items())

  def gather_resources(self):
    self.resource_gathering_attempts += 1
    # Move towards known resource locations if available
    if self.resource_locations:
      closest_resource, distance = min(self.resource_locations.items(), key=lambda x: x[1])
      direction = self.get_move_direction(closest_resource)
      if direction:
        return direction
    # Random exploration for resource discovery
    return self.explore()

  def explore(self):
    # Check for loop detection and break out if stuck
    if len(set(self.action_history)) == 1 and self.resource_gathering_attempts > 10:
      return self.random_action()
    # Explore randomly with some bias towards unexplored areas
    return self.biased_random_action()

  def get_move_direction(self, target):
    dx, dy = target[0] - self.player.pos[0], target[1] - self.player.pos
