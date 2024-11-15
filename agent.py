import random
import chess
from piece_square_tables import piece_square_table_score
from pawn_shield_storm import eval_pawn_storm
from typing import Dict
from collections import defaultdict

#an agent that moves randommly
def random_agent(BOARD):
    return random.choice(list(BOARD.legal_moves))

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
def initialize_piece_count(board):
    piece_count = defaultdict(lambda: 0)
    for piece in board.piece_map().values():
        piece_count[str(piece)] += 1

    return piece_count

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

def eval_piece_count(piece_count):
    score = 0
    for piece, count in piece_count.items():
        score += count * scoring[piece]
    return score

def featureExtractor(piece_count, board):
    return {
        "piece_count": eval_piece_count(piece_count),
        "pawn_storm": eval_pawn_storm(board),
        "piece_square": piece_square_table_score(board, piece_count)
    }

weights = {
    "piece_count": 1.0,
    "pawn_storm": 0,
    "piece_square": 0,
}

# simple evaluation function
def eval_board(piece_count, board):
    return dotProduct(featureExtractor(piece_count, board), weights)


def min_maxN(board, depth, alpha, beta, piece_count):
    if (board.is_stalemate() or board.is_insufficient_material()):
        return (0, None)
    if (board.is_checkmate()):
        score = float('inf') if (board.outcome().winner == chess.WHITE) else float('-inf')
        return (score, None)
    if (depth == 0):
        return (eval_board(piece_count, board), None)
    
    moves = list(board.legal_moves)
    scores = []

    for move in moves:
        # if the move is a capture, decrement the count of the captured piece
        was_capture = False
        if board.is_capture(move):
            captured_piece = str(board.piece_at(move.to_square))
            if (captured_piece == 'None'):
                print(f"MOVE TO: {move.to_square}")
                print(f"FEN: {board.fen()}")
            piece_count[captured_piece] -= 1
            was_capture = True

        board.push(move)

        # recursive call delegating to the other player
        score, _ = min_maxN(board,depth-1, alpha, beta, piece_count)

        # reset board and piece count
        board.pop()
        if was_capture:
            piece_count[captured_piece] += 1

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
        
# a simple wrapper function as the display only gives one imput , BOARD
def play_min_maxN(board):
    N=2*2 # depth 2 but multiply by 2 to ensure both players play per depth
    piece_count = initialize_piece_count(board)
    return min_maxN(board,N, float('-inf'), float('inf'), piece_count)[1]
