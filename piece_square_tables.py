import chess

# Piece-square tables for opening and endgame stages
# Opening tables

WEIGHT = 0.1

opening_table = {
    'P': [0, 0, 0, 0, 0, 0, 0, 0,
          5, 10, 10, -20, -20, 10, 10, 5,
          5, -5, -10, 0, 0, -10, -5, 5,
          0, 0, 0, 20, 20, 0, 0, 0,
          5, 5, 10, 25, 25, 10, 5, 5,
          10, 10, 20, 30, 30, 20, 10, 10,
          50, 50, 50, 50, 50, 50, 50, 50,
          0, 0, 0, 0, 0, 0, 0, 0],
    
    'N': [-50, -40, -30, -30, -30, -30, -40, -50,
          -40, -20, 0, 5, 5, 0, -20, -40,
          -30, 5, 10, 15, 15, 10, 5, -30,
          -30, 0, 15, 20, 20, 15, 0, -30,
          -30, 5, 15, 20, 20, 15, 5, -30,
          -30, 0, 10, 15, 15, 10, 0, -30,
          -40, -20, 0, 0, 0, 0, -20, -40,
          -50, -40, -30, -30, -30, -30, -40, -50],

    'B': [-20, -10, -10, -10, -10, -10, -10, -20,
          -10, 5, 0, 0, 0, 0, 5, -10,
          -10, 10, 10, 10, 10, 10, 10, -10,
          -10, 0, 10, 10, 10, 10, 0, -10,
          -10, 5, 5, 10, 10, 5, 5, -10,
          -10, 0, 5, 10, 10, 5, 0, -10,
          -10, 0, 0, 0, 0, 0, 0, -10,
          -20, -10, -10, -10, -10, -10, -10, -20],

    'R': [0, 0, 0, 0, 0, 0, 0, 0,
          5, 10, 10, 10, 10, 10, 10, 5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          0, 0, 0, 5, 5, 0, 0, 0],

    'Q': [-20, -10, -10, -5, -5, -10, -10, -20,
          -10, 0, 0, 0, 0, 0, 0, -10,
          -10, 0, 5, 5, 5, 5, 0, -10,
          -5, 0, 5, 5, 5, 5, 0, -5,
          0, 0, 5, 5, 5, 5, 0, -5,
          -10, 5, 5, 5, 5, 5, 0, -10,
          -10, 0, 5, 0, 0, 0, 0, -10,
          -20, -10, -10, -5, -5, -10, -10, -20],

    'K': [-30, -40, -40, -50, -50, -40, -40, -30,
          -30, -40, -40, -50, -50, -40, -40, -30,
          -30, -40, -40, -50, -50, -40, -40, -30,
          -30, -40, -40, -50, -50, -40, -40, -30,
          -20, -30, -30, -40, -40, -30, -30, -20,
          -10, -20, -20, -20, -20, -20, -20, -10,
          20, 20, 0, 0, 0, 0, 20, 20,
          20, 30, 10, 0, 0, 10, 30, 20]
}

# Endgame tables
endgame_table = {
    'P': opening_table['P'],  # Typically the same for simplicity
    'N': opening_table['N'],  # Often similar or the same for endgame
    'B': opening_table['B'],  # Bishops retain similar tables
    'R': [0, 0, 0, 0, 0, 0, 0, 0,
          5, 10, 10, 10, 10, 10, 10, 5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          -5, 0, 0, 0, 0, 0, 0, -5,
          0, 0, 0, 5, 5, 0, 0, 0],

    'Q': opening_table['Q'],  # Often similar across game phases
    'K': [20, 30, 10, 0, 0, 10, 30, 20,
          20, 20, 0, 0, 0, 0, 20, 20,
          -10, -20, -20, -20, -20, -20, -20, -10,
          -20, -30, -30, -40, -40, -30, -30, -20,
          -30, -40, -40, -50, -50, -40, -40, -30,
          -30, -40, -40, -50, -50, -40, -40, -30,
          -30, -40, -40, -50, -50, -40, -40, -30,
          -30, -40, -40, -50, -50, -40, -40, -30]
}

def game_phase(piece_count):
    # Define game phase values based on piece count for interpolation
    phase = sum(piece_count.get(piece, 0) for piece in ['P', 'N', 'B', 'R', 'Q'])
    return min(1, max(0, (24 - phase) / 24))  # Normalize between 0 and 1

def piece_square_table_score(board, piece_count):
    phase = game_phase(piece_count)
    score = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            symbol = piece.symbol().upper()
            position_score = (1 - phase) * opening_table[symbol][square] + phase * endgame_table[symbol][square]
            if piece.color == chess.WHITE:
                score += WEIGHT * position_score
            else:
                score -= WEIGHT * position_score

    return score