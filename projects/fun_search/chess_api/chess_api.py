import argparse
from stockfish import Stockfish
import chess

#from openelm.environments.chess_api.reward import Reward 


# TODO: 
# - see which rewards correlate with each other 
# - are there any combinations of the heuristics (e.g. weighted sums) that correlate with expert()?

STOCKFISH_PATH = "/mnt/c/Users/alexd/Projects/stockfish-ubuntu-x86-64-modern/stockfish/stockfish-ubuntu-x86-64-modern"

class Game:
    def __init__(self, 
                 stockfish_path,):
        self.board = chess.Board()
        self.moves = []
        self.winner = None
        self.stockfish = Stockfish(stockfish_path)
        self.player2_difficulty = 1  # Default difficulty level
        #self.reward = Reward(stockfish_path=stockfish_path).expert()

    def get_actions():
        pass

    def reset(self):
        # TODO: rename to `reset`
        self.board.reset()
        self.moves.clear()
        self.winner = None
        self.stockfish.set_position([])  # Reset Stockfish to the initial position
        return self.board, None  # observation, __

    def step(self, move_str):
        """
        return 
            observation: chess.Board
            reward: float (in [-1,1])
            terminated: bool 
            _: None 
            _: None 
        """
        # TODO: output must be 
        # observation, reward, terminated, _, _ 
        # see rl_env.py fitness()
        # TODO: define self.signal function to 
        # assign rewards
        try:
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.moves.append(move)
                if self.board.is_checkmate():
                    self.winner = 'Player 1' if self.board.turn == chess.BLACK else 'Player 2'
                    return 'checkmate'
                # After Player 1's move, call player2 for Stockfish's move
                if self.board.turn == chess.BLACK:  # Check if it's Player 2's turn
                    self.player2()
                    rw= self.reward(self.board)
                    return self.board, rw, False, None, None
            else:
                return self.board, -1, True, None, None
        except ValueError:
            return self.board, -1, True, None, None


    def player2(self):
        self.stockfish.set_skill_level(self.player2_difficulty)
        self.stockfish.set_position([move.uci() for move in self.moves])
        move_str = self.stockfish.get_best_move()
        if move_str:
            move = chess.Move.from_uci(move_str)
            self.board.push(move)
            self.moves.append(move)
            if self.board.is_checkmate():
                self.winner = 'Player 2'
                return 'checkmate'
            return move_str
        else:
            return 'no valid moves'

    def is_game_over(self):
        return self.board.is_game_over()

    def get_winner(self):
        return self.winner

    def current_board_state(self):
        return str(self.board)

    def human_player(self, move_str):
        try:
            move = chess.Move.from_uci(move_str)
            if move in self.board.legal_moves:
                self.board.push(move)
                self.moves.append(move)
                if self.board.is_checkmate():
                    self.winner = 'Human'
                    return 'checkmate'
                return 'move accepted'
            else:
                return 'illegal move'
        except ValueError:
            return 'invalid move format'
        
    # Add a method to check for turn limit
    def is_turn_limit_reached(self, limit):
        return len(self.moves) >= limit * 2  # Each full turn consists of two moves

if __name__ == "__main__":
    # Initialize the game with the path to your Stockfish binary
    game = Game(STOCKFISH_PATH)
    game.reset()

    turn_limit = 10  # Set the turn limit as desired

    while not game.is_game_over() and not game.is_turn_limit_reached(turn_limit):
        print(game.current_board_state())
        if game.board.turn == chess.WHITE:  # Human's turn
            move = input("Enter your move (in UCI format): ")
            move_result = game.human_player(move)
            print(move_result)
        else:  # Stockfish's turn
            move = game.player2()
            move_result = game.step(move)
            print("Stockfish move:", move)
            print("Result: ", move_result)

        if move_result == 'checkmate':
            break

    print("Game Over. Winner:", game.get_winner())