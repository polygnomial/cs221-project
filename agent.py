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

# initialize and return a piece count dictionary
def initialize_piece_count(board: chess.Board) -> List[int]:
    piece_count = [0 for _ in range(12)]
    for piece in board.piece_map().values():
        piece_count[get_piece_index(piece)] += 1
    return piece_count

class Agent():
    def __init__(self):
        self.piece_count = None
        self.board = None
    
    def initialize(self, board: chess.Board):
        self.piece_count = initialize_piece_count(board)
        self.board = board

    def name(self) -> str:
        raise Exception("Not yet implemented")

    def get_move(self):
        raise Exception("Not yet implemented")
    
class RandomAgent(Agent):

    def name(self) -> str:
        return "random_agent"

    def get_move(self):
        return random.choice(list(self.board.legal_moves))

class MiniMaxAgent(Agent):
    def __init__(self, depth: int):
        self.depth = depth
        self.weights =  {
            "piece_count": 1.0,
            "pawn_storm": 0,
            "piece_square": 0,
        }

    def featureExtractor(self, piece_count: List[int], board: chess.Board):
        return {
            "piece_count": eval_piece_count(self.piece_count),
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

            # reset board and piece count
            if captured_piece is not None:
                piece_count[piece_indices[captured_piece]] += 1
            board.pop()

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

    def name(self) -> str:
        return "minimax_agent"

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
    def __init__(self, depth: int):
        super().__init__(depth)
        self.weights["piece_square"] = 1

    def name(self) -> str:
        return super().name() + "_with_piece_square_tables"

# def iterative_deepening():
#     depth = 2
#     while(True):
#         min_maxN(board,N, lambda: eval_piece_count(piece_count), float('-inf'), float('inf'), piece_count)[1]
#         depth += 2
