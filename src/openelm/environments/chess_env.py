import gymnasium as gym
from gymnasium import spaces
import chess
from stockfish import Stockfish
from typing import List
from copy import deepcopy
from enum import Enum
import numpy as np


STOCKFISH_PATH = "/storage/coda1/p-wliao60/0/ahavrilla3/alex/repos/OpenELM/stockfish/stockfish"


class EnvMode(Enum):
    GROUND = 1
    IMAGINED = 2

    @classmethod
    def from_string(cls, name):
        name_to_val = {val.name: val for val in cls}
        if name_to_val.get(name.upper(), None):
            return name_to_val[name.upper()]
        else:
            raise ValueError(f"Unknown name: {name}!!!")


class RandomStockfish:
    def __init__(self, board):
        self.board = board

    def get_best_move(self):
        move = np.random.choice(list(self.board.legal_moves))
        return move.uci()

    def set_position(self, moves):
        pass


class ChessEnv:
    """ 
    Implements chess gym environment for white player. Black player is stockfish with a preset difficulty level.
    """
    FPS = 50
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": FPS,
    }

    def __init__(self,
                 stockfish=None,
                 stockfish_skill: int=1,
                 render_mode=None,
                 mode="ground",
                 board=None,
                 moves=None,):
        """ 
        + stockfish_skill: Difficulty of stockfish opponent.
            - 1 -> 1350 ELO
        """
        super(ChessEnv, self).__init__()
        self.mode = EnvMode.from_string(mode)
        self.board = chess.Board() if not board else board
        self.moves: List[chess.Move] = [] if not moves else moves
        self.stockfish_skill = stockfish_skill
        if stockfish is None and self.mode == EnvMode.GROUND:
            self.stockfish: Stockfish = Stockfish(STOCKFISH_PATH)
            self.stockfish.set_skill_level(stockfish_skill)
        elif self.mode == EnvMode.GROUND:
            self.stockfish = stockfish
            self.stockfish.set_skill_level(stockfish_skill)
        else:
            self.stockfish = RandomStockfish(self.board)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))  # Number of legal moves
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 6), dtype=int)  # 8x8 board with 6 piece types
        self.render_mode = render_mode

    def reset(self, seed):
        self.board.reset()
        self.moves.clear()
        self.stockfish.set_position([])  # Reset Stockfish to the initial position
        info = dict()
        return self._get_observation(), info
    
    def deepcopy(self):
        copy_env = ChessEnv(stockfish_skill=self.stockfish_skill, 
                            mode="imagined",
                            board=self.board,
                            moves=self.moves,)
        copy_env.num_ground_moves = len(self.moves)
        return copy_env

    def restore(self):
        while len(self.moves) > self.num_ground_moves:
            self.moves.pop()
            self.board.pop()
    
    def get_actions(self):
        return self.board.legal_moves

    def step(self, action: chess.Move):
        # Convert action index to a chess move
        legal_moves = list(self.board.legal_moves)
        if action in legal_moves:
            self.board.push(action)
            self.moves.append(action)
            done = self.board.is_game_over()
            info = dict()
            truncated = False
            if not done:
                # Make move by opponent
                self.stockfish.set_position([move.uci() for move in self.moves])
                stockfish_move = self.stockfish.get_best_move()
                stockfish_move = chess.Move.from_uci(stockfish_move)
                self.board.push(stockfish_move)
                self.moves.append(stockfish_move)
                done = self.board.is_game_over()
                reward = -int(self.board.is_checkmate())
                return self._get_observation(), reward, done, truncated, info
            else:
                reward = int(self.board.is_checkmate())
                return self._get_observation(), reward, done, truncated, info
        else:
            done = True
            truncated = True
            reward = -1
            info = {'reason': 'invalid move'}
            return self._get_observation(), reward, done, truncated, info

    def render(self):
        # Implement rendering (e.g., print the board state)
        if self.render_mode == "human":
            print(self.board)
            print("------------------------")

    def _get_observation(self):
        # Convert the board to a matrix or other representation suitable for your observation space
        return self.board