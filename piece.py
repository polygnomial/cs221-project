PAWN_VALUE = 100
KNIGHT_VALUE = 200
BISHOP_VALUE = 320
ROOK_VALUE = 500
QUEEN_VALUE = 900

def get_piece_value(piece_type: int):
    if (piece_type == 1):
        return PAWN_VALUE
    elif (piece_type == 2):
        return KNIGHT_VALUE
    elif (piece_type == 3):
        return BISHOP_VALUE
    elif (piece_type == 4):
        return ROOK_VALUE
    elif (piece_type == 5):
        return QUEEN_VALUE
    return 1000000
