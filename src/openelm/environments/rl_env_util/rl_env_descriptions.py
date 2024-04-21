envs = {
    "MiniGrid-UnlockPickup-v0": dict(
        task_description="""The agent has to pick up a box which is placed in another room, behind a locked door. This environment can be solved without relying on language.
""",
        observation_description="""You can see a (7, 7) square of tiles in the direction you are facing and your inventory of item. \
Formally \
- observation['agent']['direction'] with 0: right, 1: down, 2: left, 3: up\n\
- observation['agent']['image'] array with shape (7, 7, 3) with each tile in the (7, 7) grid encoded as the triple (object: int, color: int, state: int) where
    - object with 0: unseen, 1: empty, 2: wall, 3: floor, 4: door, 5: key, 6: ball, 7: box, 8: goal, 9: lava
    - color with 0: red, 1: green, 2: blue, 3: purple, 4: yellow, 5: grey
    - state with 0: door open, 1: door closed, 2: door locked
- observation['inv']: list[int] contains any object being held and is empty otherwise 
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
+0.1 for picking up the key for the first time. \
+0.2 for opening the door \
+0.1 for going through the door \
+0.1 for putting down the key after opening the door.
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

    "crafter": dict(
        task_description="",
        observation_description="",
        action_description="",
        reward_description="",
        action_exemplar="",
    ),
}