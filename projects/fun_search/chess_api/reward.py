"""
Reward function for chess game. 
This is for comparing with the value function that the model 
learns. They can also be used as seed functions. 
"""
import math 

import chess
import stockfish 


class Reward: 
    def __init__(self, move: chess.Move, board: chess.Board):
        """
        each reward function should return a float between -1 and 1
        """
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # The king's value is not typically counted in material balance
        }
        self.piece_counts = {
            chess.PAWN: 8,
            chess.KNIGHT: 2,
            chess.BISHOP: 2,
            chess.ROOK: 2,
            chess.QUEEN: 1,
            chess.KING: 1
        }
        self.total_value = sum(self.piece_values[piece_type] * count for piece_type, count in self.piece_counts.items()) # 39 
        self.move = move
        self.board = board

    
    def capture(self): 
        """
        Reward for capturing pieces.
        """
        # Check if the move is a capture
        if self.board.is_capture(self.move):
            captured_piece = self.board.piece_at(self.move.to_square)
            if captured_piece:
                # Reward is proportional to the value of the captured piece
                return self.piece_values[captured_piece.piece_type] / self.total_value
        return 0

    def mobility(self):
        """
        Reward for number of legal moves.
        """
        # Count legal moves before and after making the move
        legal_moves_before = self.board.legal_moves.count()
        self.board.push(self.move)
        legal_moves_after = self.board.legal_moves.count()
        self.board.pop()

        # Reward is based on the change in mobility
        mobility_change = legal_moves_after - legal_moves_before
        max_mobility = len(list(self.board.legal_moves))

        return mobility_change / max_mobility

    def king_safety(self):
        """
        Reward for king safety.
        """
        # Implement a simple king safety evaluation
        # For instance, count the attacking pieces around the king
        # This is a placeholder implementation, adjust as needed
        king_square = self.board.king(self.board.turn)
        attackers = self.board.attackers(not self.board.turn, king_square)

        return -len(attackers) / 8  # Normalize based on the max number of possible attackers

    def center_control(self):
        """
        Reward for controlling the center.
        """
        # Evaluate control of center squares: D4, D5, E4, E5
        center_squares = [chess.D4, chess.D5, chess.E4, chess.E5]
        control_score = 0

        for square in center_squares:
            if self.board.is_attacked_by(self.board.turn, square):
                control_score += 1

        return control_score / len(center_squares)

    def reward(self):
        """
        Overall reward for the current board state.
        """
        # Combine various aspects to calculate overall reward
        # Adjust weights as necessary
        return max(self.capture(),  
                self.mobility(),  
                self.king_safety(),  
                self.center_control())