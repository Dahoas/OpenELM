import gymnasium as gym
from gymnasium import spaces
import chess
from stockfish import Stockfish

from chess_api.reward import Reward 

STOCKFISH_PATH = "/usr/local/bin/stockfish"

class ChessEnv(gym.Env):
    def __init__(self):
        super(ChessEnv, self).__init__()
        self.board = chess.Board()
        self.moves = []
        self.winner = None
        self.stockfish = Stockfish(STOCKFISH_PATH)
        self.stockfish.set_skill_level(1)  # Default difficulty level
        self.reward = Reward(stockfish_path=STOCKFISH_PATH).expert()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(len(list(self.board.legal_moves)))  # Number of legal moves
        self.observation_space = spaces.Box(low=0, high=1, shape=(8, 8, 6), dtype=int)  # 8x8 board with 6 piece types

    def reset(self):
        self.board.reset()
        self.moves.clear()
        self.winner = None
        self.stockfish.set_position([])  # Reset Stockfish to the initial position
        return self._get_observation()

    def step(self, action):
        # Convert action index to a chess move
        legal_moves = list(self.board.legal_moves)
        try:
            move = legal_moves[action]
            self.board.push(move)
            self.moves.append(move)
        except IndexError:
            return self._get_observation(), -1, True, {'reason': 'invalid move'}

        done = self.board.is_game_over()
        reward = self.reward(self.board)
        info = {}
        return self._get_observation(), reward, done, info

    def render(self, mode='human'):
        # Implement rendering (e.g., print the board state)
        print(self.board)

    def _get_observation(self):
        # Convert the board to a matrix or other representation suitable for your observation space
        return self._board_to_matrix()

    def _board_to_matrix(self):
        # Implement the logic to convert the chess board to a matrix or another suitable format
        return list(map(lambda x: x.split(" "), str(board).split("\n")))
        

# Register the environment
gym.register(
    id='chess',
    entry_point='chess_env:ChessEnv',  # Replace 'your_module' with the actual module where `ChessEnv` is defined
)
