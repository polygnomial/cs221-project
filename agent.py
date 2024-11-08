import random
from copy import deepcopy
import chess

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
#simple evaluation function
def eval_board(board):
    score = 0
    pieces = board.piece_map()
    for key in pieces:
        score += scoring[str(pieces[key])]

    return score

def min_maxN(board, depth, alpha, beta):
    if (board.is_stalemate() or board.is_insufficient_material()):
        return (0, None)
    if (board.is_checkmate()):
        score = float('inf') if (board.outcome().winner == chess.WHITE) else float('-inf')
        return (score, None)
    if (depth == 0):
        return (eval_board(board), None)
    
    moves = list(board.legal_moves)
    scores = []

    for move in moves:
        temp = deepcopy(board)
        # push our move and then delegate to the adversary
        temp.push(move)
        score, _ = min_maxN(temp,depth-1, alpha, beta)
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
    return min_maxN(board,N, float('-inf'), float('inf'))[1]


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