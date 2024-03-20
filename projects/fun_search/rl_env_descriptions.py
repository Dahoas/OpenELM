envs = {
    "chess": dict( 
        task_description="""
        You are a chess world champion. Win the chess game. You are playing white. You have no time constraints.  
""", 
        observation_description="""
        observation: chess.Boards() object from the python-chess library. It has the following attributes which may be useful:
        
        move_stack: List[Move]
        The move stack. Use Board.push(), Board.pop(), Board.peek() and Board.clear_stack() for manipulation.

        propertylegal_moves: LegalMoveGenerator
        A dynamic list of legal moves.
        Note: To use this as a list you will need to do list(board.legal_moves)

        import chess

        board = chess.Board()
        board.legal_moves.count()
        20
        bool(board.legal_moves)
        True
        move = chess.Move.from_uci("g1f3")
        move in board.legal_moves
        True
        Wraps generate_legal_moves() and is_legal().


        checkers()→ SquareSet[source]
        Gets the pieces currently giving check.

        Returns a set of squares.


        is_check()→ bool[source]
        Tests if the current side to move is in check.

        gives_check(move: Move)→ bool[source]
        Probes if the given move would put the opponent in check. The move must be at least pseudo-legal.


        is_checkmate()→ bool[source]
        Checks if the current position is a checkmate.

        is_stalemate()→ bool[source]
        Checks if the current position is a stalemate.

        is_insufficient_material()→ bool[source]
        Checks if neither side has sufficient winning material (has_insufficient_material()).

        has_insufficient_material(color: chess.Color)→ bool[source]
        Checks if color has insufficient winning material.

        This is guaranteed to return False if color can still win the game.

        The converse does not necessarily hold: The implementation only looks at the material, including the colors of bishops, but not considering piece positions. So fortress positions or positions with forced lines may return False, even though there is no possible winning line.


        find_move(from_square: chess.Square, to_square: chess.Square, promotion: chess.PieceType | None = None)→ Move[source]
        Finds a matching legal move for an origin square, a target square, and an optional promotion piece type.

        For pawn moves to the backrank, the promotion piece type defaults to chess.QUEEN, unless otherwise specified.

        Castling moves are normalized to king moves by two steps, except in Chess960.

        Raises
        :
        IllegalMoveError if no matching legal move is found.
""", 
        api_description="""
        None
""", 
        action_description="""
        None
""",
        reward_description="""
        None
""",
        action_exemplar="""
        None
"""
    ), 

    "Blackjack-v1": dict(
        task_description="""Win the blackjack hand. Each round will be separate, independent of the ones before.
""",
        observation_description="""observation: Tuple[int, int, bool] where:\n\
observation[0] = The sum of your cards\n\
observation[1] = The dealer's showing card sum.\n\
observation[2] = True if you have an Ace, False otherwise\n\
""",
     action_description="""action: str \n\
chess notation for player1's move. \n\
""",
        reward_description="""A reward of between -1 and 1 is given for each move. Takes into account: number of pieces captured, piece mobility, pawn structure
         , king safety, control of center of board, how well balanced 
           the pieces are around the board.  \n\
""",
        action_exemplar="""\
- a_1 = "e2e4"  # Player 1 moves pawn from e2 to e4
- a_2 = "e7e5"  # Player 2 moves pawn from e7 to e5
- a_3 = "g1f3"  # Player 1 moves knight from g1 to f3
- a_4 = "b8c6"  # Player 2 moves knight from b8 to c6
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
+0.1 for picking up the key for the first time. +0.2 for opening the door, and +0.1 for going through the door.
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