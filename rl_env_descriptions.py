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
observation['image'][2] to the left and observation['image'][4] to the right and observation['image'][3][5] forward
""",
        action_description="""action: int such that\n\
- 0: turn left\n\
- 1: turn right\n\
- 2: move forward\n\
- 3: pickup item\n
""",
        reward_description="""A reward of ‘1 - 0.9 * (step_count / max_steps)’ is given for success, and ‘0’ for failure.
""",
        action_exemplar="""\
- a_1 = 1  # I don't see anything so turn right
- a_2 = 2  # I see the key to my forward left and the door to my right so I walk toward both
- a_3 = 2  # Walk forward again toward key
- a_4 = 0  # Turn toward key
- a_5 = 3  # pickup key
- a_6 = 1  # Turn right to pickup ball to my right
- a_7 = 2  # Walk toward ball
- a_8 = 3  # pickup ball in front of me
- a_9 = 2  # Walk forward towards door
- a_10 = 2  # Walk through door with key
- a_11 = 2  # Walk forward
- a_12 = 1  # I see the chest to my right so I turn right
- a_13 = 2  # Walk toward chest
- a_14 = 3  # Pickup chest
""",
    ),
}