import gymnasium as gym
from gymnasium import spaces
import chess
from stockfish import Stockfish
from typing import List


STOCKFISH_PATH = "/mnt/c/Users/alexd/Projects/stockfish-ubuntu-x86-64-modern/stockfish/stockfish-ubuntu-x86-64-modern"

class ChessEnv(gym.Env):
    """ 
    Implements chess gym environment for white player. Black player is stockfish with a preset difficulty level.
    """
    def __init__(self,
                 stockfish_skill: int=1,):
        """ 
        + stockfish_skill: Difficulty of stockfish opponent.
            - 1 -> 1350 ELO
        """
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.moves: List[chess.Move] = []
        self.stockfish: Stockfish = Stockfish(STOCKFISH_PATH)
        self.stockfish.set_skill_level(stockfish_skill)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))  # Number of legal moves
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 6), dtype=int)  # 8x8 board with 6 piece types

    def reset(self, seed):
        self.board.reset()
        self.moves.clear()
        self.stockfish.set_position([])  # Reset Stockfish to the initial position
        info = dict()
        return self._get_observation(), info
    
    def deepcopy(self):
        copy_env = ChessEnv()
        copy_env.board = self.board
        copy_env.moves = self.moves
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
            if done:
                reward = int(self.board.is_checkmate())
                return self._get_observation(), reward, done, info, info
            else:
                # Make move by opponent
                self.stockfish.set_position([move.uci() for move in self.moves])
                stockfish_move = self.stockfish.get_best_move()
                stockfish_move = chess.Move.from_uci(stockfish_move)
                self.board.push(stockfish_move)
                self.moves.append(stockfish_move)
                done = self.board.is_game_over()
                reward = -int(self.board.is_checkmate())
                return self._get_observation(), reward, done, info, info
        else:
            done = True
            reward = -1
            info = {'reason': 'invalid move'}
            return self._get_observation(), reward, done, info, info

    def render(self, mode='human'):
        # Implement rendering (e.g., print the board state)
        print(self.board)

    def _get_observation(self):
        # Convert the board to a matrix or other representation suitable for your observation space
        return self.board