import chess
import numpy as np

# Piece-square tables for opening and endgame stages
# Opening tables

WEIGHT = 0.01

opening_table = {
    'P': [
        0,   0,   0,   0,   0,   0,   0,   0,
        50,  50,  50,  50,  50,  50,  50,  50,
        10,  10,  20,  30,  30,  20,  10,  10,
        5,   5,  10,  25,  25,  10,   5,   5,
        0,   0,   0,  20,  20,   0,   0,   0,
        5,  -5, -10,   0,   0, -10,  -5,   5,
        5,  10,  10, -20, -20,  10,  10,   5,
        0,   0,   0,   0,   0,   0,   0,   0
    ],
    
    'N': [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],

    'B': [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],

    'R': [
        0, 0, 0, 0, 0, 0, 0, 0,
        5, 10, 10, 10, 10, 10, 10, 5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        -5, 0, 0, 0, 0, 0, 0, -5,
        0, 0, 0, 5, 5, 0, 0, 0
    ],

    'Q': [

        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,   0,  5,  5,  5,  5,  0, -5,
         0,   0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],

    'K': [
        -80, -70, -70, -70, -70, -70, -70, -80, 
        -60, -60, -60, -60, -60, -60, -60, -60, 
        -40, -50, -50, -60, -60, -50, -50, -40, 
        -30, -40, -40, -50, -50, -40, -40, -30, 
        -20, -30, -30, -40, -40, -30, -30, -20, 
        -10, -20, -20, -20, -20, -20, -20, -10, 
        20,  20,  -5,  -5,  -5,  -5,  20,  20, 
        20,  30,  10,   0,   0,  10,  30,  20
    ]
}

# Endgame tables
endgame_table = {
    'P': [
         0,   0,   0,   0,   0,   0,   0,   0,
        80,  80,  80,  80,  80,  80,  80,  80,
        50,  50,  50,  50,  50,  50,  50,  50,
        30,  30,  30,  30,  30,  30,  30,  30,
        20,  20,  20,  20,  20,  20,  20,  20,
        10,  10,  10,  10,  10,  10,  10,  10,
        10,  10,  10,  10,  10,  10,  10,  10,
         0,   0,   0,   0,   0,   0,   0,   0
    ],
    'N': opening_table['N'],  # Often similar or the same for endgame
    'B': opening_table['B'],  # Bishops retain similar tables
    'R': opening_table['R'],
    'Q': opening_table['Q'],  # Often similar across game phases
    'K': [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -5,   0,   5,   5,   5,   5,   0,  -5,
        -10, -5,   20,  30,  30,  20,  -5, -10,
        -15, -10,  35,  45,  45,  35, -10, -15,
        -20, -15,  30,  40,  40,  30, -15, -20,
        -25, -20,  20,  25,  25,  20, -20, -25,
        -30, -25,   0,   0,   0,   0, -25, -30,
        -50, -30, -30, -30, -30, -30, -30, -50
    ]
}

orig_keys = list(opening_table.keys())
for key in orig_keys:
    black_key = key.lower()
    opening_table[black_key] = sum([opening_table[key][i:i+8] for i in range(0, 64, 8)][::-1], [])
    endgame_table[black_key] = sum([endgame_table[key][i:i+8] for i in range(0, 64, 8)][::-1], [])

def game_phase(piece_count, player):

    # Endgame Transition (0->1)
    queenEndgameWeight = 45
    rookEndgameWeight = 20
    bishopEndgameWeight = 10
    knightEndgameWeight = 10

    # Weights are: p, n, b, r, q, k
    transition_weights = [0, 10, 10, 20, 45, 0]

    endgameStartWeight = 2 * rookEndgameWeight + 2 * bishopEndgameWeight + 2 * knightEndgameWeight + queenEndgameWeight
    if player == chess.WHITE:
        endgameWeightSum = np.dot(piece_count[6:], transition_weights)
    else:
        endgameWeightSum = np.dot(piece_count[:6], transition_weights)
    endgameT = 1 - min(1, endgameWeightSum / endgameStartWeight)
    return endgameT
    

def piece_square_table_score(board, piece_count):
    endgameT = game_phase(piece_count, board.turn)
    score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol()
            position_score = (1 - endgameT) * opening_table[symbol][square]
            position_score += endgameT * endgame_table[symbol][square]
            score += WEIGHT * position_score
            # I think we don't need the below because our 
            # piece square tables have negative entries
            # if piece.color == chess.WHITE:
            #     score += WEIGHT * position_score
            # else:
            #     score -= WEIGHT * position_score

    return score