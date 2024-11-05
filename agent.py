import random
from copy import deepcopy

#an agent that moves randommly
def random_agent(BOARD):
    return random.choice(list(BOARD.legal_moves))

scoring= {'p': -1,
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
def eval_board(BOARD):
    score = 0
    pieces = BOARD.piece_map()
    for key in pieces:
        score += scoring[str(pieces[key])]

    return score

def min_maxN(BOARD,N):
    moves = list(BOARD.legal_moves)
    scores = []

    for move in moves:
        temp = deepcopy(BOARD)
        temp.push(move)

        if N>1:
            temp_best_move = min_maxN(temp,N-1)
            temp.push(temp_best_move)

        scores.append(eval_board(temp))

    if BOARD.turn == True:
        best_move = moves[scores.index(max(scores))]
    else:
        best_move = moves[scores.index(min(scores))]

    return best_move
        
# a simple wrapper function as the display only gives one imput , BOARD
def play_min_maxN(BOARD):
    N=4
    return min_maxN(BOARD,N)


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