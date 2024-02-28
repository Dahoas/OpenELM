"""
value function for chess game. 
THis is for comparing with the value function that the model 
learns. They can also be used as seed functions. 
"""
import math 

import chess
import stockfish 


class Value: 
    def __init__(self, stockfish_path=None):
        """
        class for evaluating chess boards
        each value function should return a float between -1 and 1
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
        self.stockfish_path = stockfish_path if stockfish_path else "/usr/local/bin/stockfish"

        

    def naive(self, obs: chess.Board) -> float: 
        """
        returns a naive value based on how many 
        of each type of piece is on the board
        """
        white_score = sum(self.piece_values[piece.piece_type] 
                        for piece in obs.piece_map().values() 
                        if piece.color == chess.WHITE)

        black_score = sum(self.piece_values[piece.piece_type] 
                        for piece in obs.piece_map().values() 
                        if piece.color == chess.BLACK)

        total_score = white_score - black_score
        normalized_score = total_score / self.total_value   # 16 is the maximum possible score for one side
        
        return normalized_score


    def expert(self, obs:chess.Board, backend="stockfish") -> float: 
        """
        wrapper for value functions for chess engines 
        """
        if backend == "stockfish": 
            return self.stockfish_value(obs)
        else: 
            raise NotImplementedError
        

    def stockfish_value(self, obs: chess.Board) -> float: 
        """
        Returns the value from stockfish.
        """
        sf = stockfish.Stockfish(self.stockfish_path)
        sf.set_fen_position(obs.fen())

        evaluation = sf.get_evaluation()

        if evaluation['type'] == 'cp':
            # Apply a sigmoid function to normalize the centipawn value to [-1, 1]
            cp_value = evaluation['value']
            score = 2 / (1 + math.exp(-cp_value / 100)) - 1  # Adjust the scaling factor as needed
        elif evaluation['type'] == 'mate':
            # If mate is inevitable, return -1 or 1 depending on the player
            score = 1 if evaluation['value'] > 0 else -1
        else:
            raise ValueError("Unknown evaluation type from sf.")

        return score
        
    def piece_mobility(self, obs: chess.Board) -> float:
        """
        value based on piece mobility: more moves available means a better position.
        We calculate mobility as the number of legal moves available for each side.
        """
        # Initialize counters for white and black mobility
        white_mobility = 0
        black_mobility = 0

        # Iterate over all legal moves and count them by color
        for move in obs.legal_moves:
            if obs.color_at(move.from_square) == chess.WHITE:
                white_mobility += 1
            else:
                black_mobility += 1

        # Calculate and return the normalized mobility score
        total_mobility = white_mobility + black_mobility
        if total_mobility > 0:  # Avoid division by zero
            return (white_mobility - black_mobility) / total_mobility
        else:
            return 0

    def pawn_structure(self, obs: chess.Board) -> float:
        """
        Evaluate pawn structure: doubled and isolated pawns are typically weaknesses.
        """
        def count_doubled_pawns(board):
            white_doubled = black_doubled = 0
            for file in chess.FILE_NAMES:
                file_squares = [chess.parse_square(f"{file}{rank}") for rank in chess.RANK_NAMES]
                white_pawns = sum(board.piece_at(square).piece_type == chess.PAWN and board.piece_at(square).color == chess.WHITE for square in file_squares if board.piece_at(square))
                black_pawns = sum(board.piece_at(square).piece_type == chess.PAWN and board.piece_at(square).color == chess.BLACK for square in file_squares if board.piece_at(square))
                
                if white_pawns > 1:
                    white_doubled += white_pawns - 1
                if black_pawns > 1:
                    black_doubled += black_pawns - 1

            return white_doubled, black_doubled

        def count_isolated_pawns(board):
            white_isolated = black_isolated = 0
            for file_index, file in enumerate(chess.FILE_NAMES):
                for rank in chess.RANK_NAMES:
                    square = chess.parse_square(f"{file}{rank}")
                    if board.piece_at(square) and board.piece_at(square).piece_type == chess.PAWN:
                        is_isolated = True
                        for adjacent_file in chess.FILE_NAMES[max(0, file_index - 1):min(7, file_index + 1) + 1]:
                            if adjacent_file != file:
                                for adjacent_rank in chess.RANK_NAMES:
                                    adjacent_square = chess.parse_square(f"{adjacent_file}{adjacent_rank}")
                                    if board.piece_at(adjacent_square) and board.piece_at(adjacent_square).piece_type == chess.PAWN and board.piece_at(square).color == board.piece_at(adjacent_square).color:
                                        is_isolated = False
                                        break
                                if not is_isolated:
                                    break
                        if is_isolated:
                            if board.piece_at(square).color == chess.WHITE:
                                white_isolated += 1
                            else:
                                black_isolated += 1
            return white_isolated, black_isolated
        
        white_doubled, black_doubled = count_doubled_pawns(obs)
        white_isolated, black_isolated = count_isolated_pawns(obs)

        # You can adjust the weights if necessary
        doubled_pawn_penalty = -0.5
        isolated_pawn_penalty = -0.5

        white_score = (white_doubled * doubled_pawn_penalty) + (white_isolated * isolated_pawn_penalty)
        black_score = (black_doubled * doubled_pawn_penalty) + (black_isolated * isolated_pawn_penalty)

        return (white_score - black_score) / 8 



    def king_safety(self, obs: chess.Board) -> float:
        """
        Evaluates the safety of the kings based on pawn shields and open files.
        This is a simplified heuristic and does not account for all aspects of king safety.
        """

        def evaluate_king_safety(board, color):
            king_square = board.king(color)
            pawn_shield_score = evaluate_pawn_shield(board, king_square, color)
            open_file_score = evaluate_open_file(board, king_square, color)

            # Combine scores. You may adjust weights.
            return pawn_shield_score + open_file_score

        def evaluate_pawn_shield(board, king_square, color):
            pawn_shield_score = 0
            pawn_positions = [-1, 0, 1]  # Check for pawns in the squares directly in front of the king
            for pawn_position in pawn_positions:
                # Check if there's a pawn in front of the king based on its color
                check_square = king_square + (8 if color == chess.WHITE else -8) + pawn_position
                if chess.square_rank(check_square) in range(8):  # Ensure the square is on the board
                    piece = board.piece_at(check_square)
                    if piece and piece.piece_type == chess.PAWN and piece.color == color:
                        pawn_shield_score += 1  # Adjust score based on your evaluation criteria

            # Normalize or adjust score as needed
            return pawn_shield_score / len(pawn_positions)  # Simple normalization

        def evaluate_open_file(board, king_square, color):
            file = chess.square_file(king_square)
            open_file_score = 0

            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN:
                    open_file_score -= 1  # Penalize if the file is not open
                    break

            # Normalize or adjust score as needed
            return open_file_score


        white_king_safety = evaluate_king_safety(obs, chess.WHITE)
        black_king_safety = evaluate_king_safety(obs, chess.BLACK)

        # Normalize and return the difference. More positive values indicate better safety for white, and vice versa.
        return (white_king_safety - black_king_safety) / 2


    def center_control(self, obs: chess.Board) -> float:
        """
        Evaluates control of the center of the board by counting how many times the central
        squares can be attacked or moved to by each side. More control = higher score.
        """
        def square_control(board, square, color):
            """
            Counts how many pieces of the given color can move to or attack the specified square.
            """
            control_count = 0
            for piece_square in board.piece_map():
                piece = board.piece_at(piece_square)
                if piece and piece.color == color:
                    # Generate moves for this piece and check if it can move to or attack the square
                    if square in list(board.attacks(piece_square)):
                        control_count += 1
            return control_count / 16

        center_squares = [chess.D4, chess.E4, chess.D5, chess.E5]
        white_control = 0
        black_control = 0

        for square in center_squares:
            # Check control for each square
            white_control += square_control(obs, square, chess.WHITE)
            black_control += square_control(obs, square, chess.BLACK)

        # Normalize or adjust scores as needed. Here, it's a simple difference.
        return (white_control - black_control)/16 

 

    def material_balance(self, obs: chess.Board) -> float:
        """
        Calculates the material balance on the board. Positive values indicate an advantage for white,
        negative values for black. Does not account for the strategic positions of pieces.
        """
        # Initialize counters
        white_material = 0
        black_material = 0

        # Iterate over all pieces on the board
        for piece in obs.piece_map().values():
            if piece.piece_type != chess.KING:  # Exclude kings from material count
                if piece.color == chess.WHITE:
                    white_material += self.piece_values[piece.piece_type]
                else:
                    black_material += self.piece_values[piece.piece_type]

        # Calculate material balance
        material_balance = white_material - black_material

        return material_balance / self.total_value  



           