envs = {
    "Blackjack-v1": dict(
        task_description="""Win the blackjack hand. Each round will be separate, independent of the ones before.
""",
        observation_description="""observation: Tuple[int, int, bool] where:\n\
observation[0] = The sum of your cards\n\
observation[1] = The dealer's showing card sum.\n\
observation[2] = True if you have an Ace, False otherwise\n\
""",
        action_description="""action: bool where:
action = True if you hit, False if you stay
""",
        reward_description="""1 if you win and 0 otherwise.
""",
        action_exemplar="""
""",
        api_list="""
""",
    ),

    "MiniGrid-BlockedUnlockPickup-v0": dict(
        task_description="""You are an agent in 2-D gridworld. The agent has to pick up a box which is placed in another room, behind a locked door. \
The door is also blocked by a ball which the agent has to move before it can unlock the door. \
Hence, the agent has to learn to move the ball, pick up the key, open the door and pick up the object in the other room.
""",
        observation_description="""You can only see a (7, 7) square of tiles in the direction you are facing. \
Formally `observation: Dict('direction': Discrete(4), 'image':  array: (7, 7, 3)))` \
where:
- observation['direction'] with 0: right, 1: down, 2: left, 3: up\n\
- observation['image'] array with shape (7, 7, 3) with each tile in the (7, 7) grid encoded as the triple (object: int, color: int, state: int) where
    - object with 0: unseen, 1: empty, 2: wall, 3: floor, 4: door, 5: key, 6: ball, 7: box, 8: goal, 9: lava
    - color with 0: red, 1: green, 2: blue, 3: purple, 4: yellow, 5: grey
    - state with 0: door open, 1: door closed, 2: door locked
Note, the agent is always located at observation['image'][3][6] with \
observation['image'][2] to the left and observation['image'][4] to the right and observation['image'][3][5] forward.
""",
        action_description="""action: int such that\n\
- 0: turn left\n\
- 1: turn right\n\
- 2: move forward, Precondition: Forward tile must be empty\n\
- 3: pickup item, Precondition: must be standing in tile adjacent to object and facing it (item must be on observation['image'][3][5]). Cannot be holding another object\n\
- 4: drop item, Precondition: Must be holding item. Tile you are facing must be empty\n\
- 5: toggle key to open door, Precondition: Must have key and be facing the door (door is on observation['image'][3][5])
""",
        reward_description="""A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.\
+0.1 for picking up the key, ball for the first time. +0.2 for opening the door.
""",
        action_exemplar="""
""",
        api_description="""\
def move_towards(target_position, next_tile):
    target_x, target_y = target_position
    if target_y < 6 and next_tile[0] == 1:
        return 2  # If target is in front of me and the next stil is empty, go forward
    else:
        if target_x < 3:
            return 0  # If target is to my left, turn left
        else:
            return 1  # If target is to my right, turn right
""",
        api_list=["move_towards"],
    ),
    "MiniGrid-UnlockPickup-v0": dict(
        task_description="""The agent has to pick up a box which is placed in another room, behind a locked door. This environment can be solved without relying on language.
""",
        observation_description="""You can only see a (7, 7) square of tiles in the direction you are facing. \
Formally `observation: Dict('direction': Discrete(4), 'image':  array: (7, 7, 3)))` \
where:
- observation['direction'] with 0: right, 1: down, 2: left, 3: up\n\
- observation['image'] array with shape (7, 7, 3) with each tile in the (7, 7) grid encoded as the triple (object: int, color: int, state: int) where
    - object with 0: unseen, 1: empty, 2: wall, 3: floor, 4: door, 5: key, 6: ball, 7: box, 8: goal, 9: lava
    - color with 0: red, 1: green, 2: blue, 3: purple, 4: yellow, 5: grey
    - state with 0: door open, 1: door closed, 2: door locked
Note, the agent is always located at observation['image'][3][6] with \
observation['image'][2] to the left and observation['image'][4] to the right and observation['image'][3][5] forward.
""",
        action_description="""action: int such that\n\
- 0: turn left\n\
- 1: turn right\n\
- 2: move forward, Precondition: Forward tile must be empty\n\
- 3: pickup item, Precondition: must be standing in tile adjacent to object and facing it (item must be on observation['image'][3][5]). Cannot be holding another object\n\
- 4: drop item, Precondition: Must be holding item. Tile you are facing must be empty\n\
- 5: toggle key to open door, Precondition: Must have key and be facing the door (door is on observation['image'][3][5])
""",
        reward_description="""A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.\
+0.1 for picking up the key for the first time. +0.2 for opening the door, +0.1 if gone through door.
""",
        action_exemplar="""
""",
        api_description="""\
def move_towards(target_position: Tuple[int, int], next_tile: Tuple[int, int, int]):\
    "\
    Move your agent toward the target position `target_position` given in (x,y) coords.
    next_tile is the tile corresponding to observation["image"][3][5] (i.e. the tile in front of you)
    "
""",
        api_list=["move_towards"],
    ),
}


minigrid_action_exemplar = """
- a_1 = 1  # I don't see anything so turn right
- a_2 = 2  # I see the key to my forward left and the door to my right so I walk toward both
- a_3 = 2  # Walk forward again toward key
- a_4 = 0  # Turn toward key
- a_5 = 3  # pickup key in the square in front of me
- a_6 = 1  # Turn right to pickup ball to my right
- a_7 = 1  # Turn right again
- a_8 = 2  # Walk toward ball
- a_9 = 3  # pickup ball in the square front of me
- a_10 = 2  # Walk forward towards door
- a_11 = 2  # Walk through door with key
- a_12 = 2  # Walk forward
- a_13 = 1  # I see the chest to my right so I turn right
- a_14 = 2  # Walk toward chest
- a_15 = 3  # Pickup chest in the square in front of me"""