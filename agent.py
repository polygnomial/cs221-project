import chess
import random
from piece_square_tables import piece_square_table_score
from pawn_shield_storm import eval_pawn_storm
from typing import Callable, Dict, List
from collections import defaultdict

piece_indices = {
    'p': 0, # Black pawn
    'n': 1, # Black knight
    'b': 2, # Black bishop
    'r': 3, # Black rook
    'q': 4, # Black queen
    'k': 5,  # Black king
    'P': 6,  # White pawn
    'N': 7,  # White knight
    'B': 8,  # White bishop
    'R': 9,  # White rook
    'Q': 10,  # White queen
    'K': 11,  # White king
}

index_pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']

scoring= {
    'p': -1, # Black pawn
    'n': -3, # Black knight
    'b': -3, # Black bishop (should be slightly more valuable than knight ideally for better evaluation)
    'r': -5, # Black rook
    'q': -9, # Black queen
    'k': 0,  # Black king
    'P': 1,  # White pawn
    'N': 3,  # White knight
    'B': 3,  # White bishop (should be slightly more valuable than knight ideally for better evaluation)  
    'R': 5,  # White rook
    'Q': 9,  # White queen
    'K': 0,  # White king
}

# initialize and return a piece count dictionary
def initialize_piece_count(board: chess.Board) -> List[int]:
    piece_count = [0 for _ in range(12)]
    for piece in board.piece_map().values():
        piece_count[get_piece_index(piece)] += 1
    return piece_count

def eval_piece_count(piece_count):
    score = 0
    for i in range(len(piece_count)):
        score += piece_count[i] * scoring[index_pieces[i]]
    return score

def get_piece_index(piece: chess.Piece):
    return piece_indices[piece.symbol()]

def get_captured_piece(board: chess.Board, move: chess.Move):
    captured_piece = str(board.piece_at(move.to_square))
    if (captured_piece != 'None'):
        return captured_piece
    
    #en passant
    idx = 0
    if (move.from_square > move.to_square): # black captures white piece
        if (move.from_square - 7 == move.to_square): # piece is to the right
            idx = move.from_square + 1
        else: # piece is to the left
            idx = move.from_square - 1
    else: # white captures black piece
        if (move.from_square + 7 == move.to_square): # piece is to the left
            idx = move.from_square - 1
        else: # piece is to the right
            idx = move.from_square + 1
    return str(board.piece_at(idx))

def dotProduct(d1: Dict, d2: Dict) -> float:
    """
    The dot product of two vectors represented as dictionaries. This function
    goes over all the keys in d2, and for each key, multiplies the corresponding
    values in d1 and d2 and adds the result to a running sum. If the key is not
    in d1, it is treated as having value 0.

    @param dict d1: a feature vector represented by a mapping from a feature (string) to a weight (float).
    @param dict d2: same as d1
    @return float: the dot product between d1 and d2
    """
    if len(d1) < len(d2):
        return dotProduct(d2, d1)
    else:
        return sum(d1.get(f, 0) * v for f, v in list(d2.items()))


class Agent():
    def __init__(self, name: str):
        self.piece_count = None
        self.board = None
        self._name = name
    
    def initialize(self, board: chess.Board):
        self.piece_count = initialize_piece_count(board)
        self.board = board

    def name(self) -> str:
        return self._name

    def get_move(self):
        raise Exception("Not yet implemented")
    
class RandomAgent(Agent):

    def get_move(self):
        return random.choice(list(self.board.legal_moves))

class MiniMaxAgent(Agent):
    def __init__(self, name, depth: int):
        super().__init__(name)
        self.depth = depth
        self.weights =  {
            "piece_count": 1.0,
            "mobility_assess": 0.2, #added mobility WRT legal moves vs. opponent
            "king_safety": 0.5, #added king safety vs. opponent
            "pawn_storm": 0,
            "piece_square": 0,
        }
    def evaluate_mobility(self, board: chess.Board):
        """
        mobility WRT legal moves; returns positive int if active has more legal moves
        """
        own_mobility = len(list(board.legal_moves)) #active player legal moves then switch to opponent
        board.push(chess.Move.null()) #null move switch turns without alteration
        opponent_mobility = len(list(board.legal_moves)) #opponent's legal moves
        board.pop() #remove null move switch to restore
        return own_mobility - opponent_mobility #positive=more mobility than opponent

    def evaluate_king_safety(self, board: chess.Board):
        """
        float positive for safer white king, negative for safe black king
        """
        white_king_safety = self._king_safety(board, chess.WHITE)
        black_king_safety = self._king_safety(board, chess.BLACK)
        return white_king_safety - black_king_safety #positive = white king safer
    def _king_safety(self, board: chess.Board, color: chess.Color):
        """
               Evaluate the safety of the king of the specified color.

        Args:
            board (chess.Board): The current board state.
            color (chess.Color): The color of the king to evaluate.

        Returns:
            float: The safety score for the king.
        """
        king_square = board.king(color) #where is king

        if king_square is None: #shouldn't happen, just to be safe
            return float('-inf') if color == chess.WHITE else float('inf')
        safety_score = 0
        safety_score += self._pawn_shield(board, king_square, color) #protect the king
        safety_score += self._open_lines_to_king(board, king_square, color) #dangerous paths of access to king
        safety_score += self._enemy_pieces_near_king(board, king_square, color) #dangerous nearby pieces
        #safety_score += self._castling_status(board, color) #castling option if needed adds safety
        safety_score += self._king_exposure(board, king_square, color) #overall exposure risk for king
        return safety_score

    def _pawn_shield(self, board: chess.Board, king_square: int, color: chess.Color):
        """
        Positive if pawns are protecting the king, otherwise negative.
        """
        shield_squares = self._get_pawn_shield_squares(king_square, color)
        pawn_shield_score = 0
        for square in shield_squares:
            if not (0 <= square <= 63):  # Check if square index is valid
                continue
            piece = board.piece_at(square)
            if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
                pawn_shield_score += 0.5
            else:
                pawn_shield_score -= 0.5  # Penalize no shield
        return pawn_shield_score

    def _get_pawn_shield_squares(self, king_square: int, color: chess.Color):
        """
        get cells comprising potential pawn shield
        """
        rank = chess.square_rank(king_square)
        file = chess.square_file(king_square)

        shield_squares = []
#add rank if self color is in front of king; o/w penalize
        if color == chess.WHITE:
            rank += 1
        else:
            rank -= 1
        for df in [-1, 0, 1]:
            f = file + df
            if 0 <= f <= 7 and 0 <= rank <= 7: #try to bound file and rank; had trouble here
                shield_squares.append(chess.square(f, rank))
        return shield_squares

    def _open_lines_to_king(self, board: chess.Board, king_square: int, color: chess.Color):
        #open files and diags to king. Might not be realistic?

        open_files_penalty = 0
        file = chess.square_file(king_square)
        friendly_pawns_in_file = any(
            board.piece_at(chess.square(file, r)) == chess.Piece(chess.PAWN, color)
            for r in range(8)
        )
        if not friendly_pawns_in_file: #penalize no friendly pawns in file
            #should add additional rules and biases for non-pawns in file.
            open_files_penalty -= 0.5
        return open_files_penalty

    def _enemy_pieces_near_king(self, board: chess.Board, king_square: int, color: chess.Color):
        enemy_color = not color
        threat_penalty = 0
        # Get the squares adjacent to the king
        king_area = chess.SquareSet(chess.BB_KING_ATTACKS[king_square]) #from 64 bit integer BB all squares to which king can move from current and convert to chess.SquareSet for iteration over squares...only squares to which king can move
        for square in king_area:
            piece = board.piece_at(square)
            if piece is not None and piece.color == enemy_color:
                threat_penalty -= 0.3
        return threat_penalty

    #def _castling_status(self, board: chess.Board, color: chess.Color):
    #    if board.is_castled(color):
    #        return 0.5  # Reward for castling
    #    else:
    #        return -0.5  # Penalty for not castling

    def _king_exposure(self, board: chess.Board, king_square: int, color: chess.Color):
        """
        penalize getting too deep into the board; probably should vary depending upon game state
        """
        exposure_penalty = 0
        rank = chess.square_rank(king_square)
        if color == chess.WHITE and rank >= 5:
            exposure_penalty -= 0.5  # Penalty for white king being too advanced
        elif color == chess.BLACK and rank <= 2:
            exposure_penalty -= 0.5  # Penalty for black king being too advanced
        return exposure_penalty



    def featureExtractor(self, piece_count: List[int], board: chess.Board):
        return {
            "piece_count": eval_piece_count(self.piece_count),
            "mobility_assess": self.evaluate_mobility(board) if self.weights.get("mobility_assess", 0) > 0 else 0, #how many legal moves availabl assuming >0
            "king_safety": self.evaluate_king_safety(board) if self.weights.get("king_safety", 0) > 0 else 0,
            "pawn_storm": eval_pawn_storm(board) if self.weights["pawn_storm"] > 0.0 else 0,
            "piece_square": piece_square_table_score(board, piece_count) if self.weights["piece_square"] > 0.0 else 0

        }

    # simple evaluation function
    def eval_board(self, board: chess.Board, piece_count: List[int]):
        return dotProduct(self.featureExtractor(piece_count, board), self.weights)

    def min_maxN(
            self,
            board: chess.Board,
            piece_count: List[int],
            depth: int,
            eval_fn: Callable[[chess.Board, List[int]], float],
            alpha: float,
            beta: float):
        if (board.is_stalemate() or board.is_insufficient_material()):
            return (0, None)
        if (board.is_checkmate()):
            score = float('inf') if (board.outcome().winner == chess.WHITE) else float('-inf')
            return (score, None)
        if (depth == 0):
            return (eval_fn(), None)
        
        moves = list(board.legal_moves)
        scores = []

        for move in moves:
            # if the move is a capture, decrement the count of the captured piece
            captured_piece = None
            if board.is_capture(move):
                captured_piece = get_captured_piece(board, move)
                piece_count[piece_indices[captured_piece]] -= 1

            board.push(move)

            # recursive call delegating to the other player
            score, _ = self.min_maxN(
                board=board,
                piece_count=piece_count,
                depth=depth - 1,
                eval_fn=eval_fn,
                alpha=alpha,
                beta=beta)

            board.pop()

            # reset board and piece count
            if captured_piece is not None:
                piece_count[piece_indices[captured_piece]] += 1

            if (board.turn == chess.WHITE): # max
                if (score >= beta): #prune
                    return (score, move)
                if (score > alpha):
                    alpha = score
            else: #min
                if (score <= alpha): #prune
                    return (score, move)
                if (score < beta):
                    beta = score
            scores.append(score)

        bestScore = max(scores) if board.turn == chess.WHITE else min(scores)
        return (bestScore, moves[scores.index(bestScore)])

    def get_move(self):
        _, move = self.min_maxN(
            board=self.board,
            piece_count=self.piece_count,
            depth=self.depth*2,
            eval_fn=lambda: self.eval_board(self.board, self.piece_count),
            alpha=float('-inf'),
            beta=float('inf'))
        return move

class MinimaxAgentWithPieceSquareTables(MiniMaxAgent):
    def __init__(self, name, depth: int):
        super().__init__(name, depth)
        self.weights["piece_square"] = 1
        self.weights["mobility_assess"] = 0.3 #mobility assessment
        self.weights["king_safety"] = 0.0 #safe king asssessment

# def iterative_deepening():
#     depth = 2
#     while(True):
#         min_maxN(board,N, lambda: eval_piece_count(piece_count), float('-inf'), float('inf'), piece_count)[1]
#         depth += 2
