import gymnasium as gym
from gymnasium import spaces
import chess
from stockfish import Stockfish
from typing import List
from copy import deepcopy


STOCKFISH_PATH = "/mnt/c/Users/alexd/Projects/stockfish-ubuntu-x86-64-modern/stockfish/stockfish-ubuntu-x86-64-modern"

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
                 stockfish_skill: int=1,
                 render_mode=None,):
        """ 
        + stockfish_skill: Difficulty of stockfish opponent.
            - 1 -> 1350 ELO
        """
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.moves: List[chess.Move] = []
        self.stockfish_skill = stockfish_skill
        self.stockfish: Stockfish = Stockfish(STOCKFISH_PATH)
        self.stockfish.set_skill_level(stockfish_skill)
        
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
        copy_env = ChessEnv(stockfish_skill=self.stockfish_skill)
        copy_env.board = deepcopy(self.board)
        copy_env.moves = deepcopy(self.moves)
        copy_env.stockfish.set_position(self.moves)
        return copy_env
    
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
            if done:
                reward = int(self.board.is_checkmate())
                return self._get_observation(), reward, done, truncated, info
            else:
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