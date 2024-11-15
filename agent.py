import random
import chess
from piece_square_tables import piece_square_table_score

#an agent that moves randommly
def random_agent(BOARD):
    return random.choice(list(BOARD.legal_moves))

scoring= {
    'p': -1,
    'n': -3,
    'b': -3,
    'r': -5,
    'q': -9,
    'k': 0,
    'P': 1,
    'N': 3,
    'B': 3,
    'R': 5,
    'Q': 9,
    'K': 0,
}


# initialize and return a piece count dictionary
def initialize_piece_count(board):
    
    piece_count = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
               'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0}

    pieces = board.piece_map()
    for piece in pieces.values():
        piece_count[str(piece)] += 1

    return piece_count

# simple evaluation function
def eval_board(piece_count, board):
    score = 0
    for piece, count in piece_count.items():
        score += count * scoring[piece]
    score += piece_square_table_score(board, piece_count)
    return score


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


# import chess.polyglot
# import chess

# BOARD = chess.Board()
# #opening book
# reader = chess.polyglot.open_reader('baron30.bin')

# #search the opening book for this game state
# opening_move = reader.get(BOARD)

# #if no move is found
# if opening_move == None:
#     move = None
# #if move is found
# else:
#     move = opening_move.move

# print(move)