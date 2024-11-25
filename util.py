import chess
from enum import Enum
import itertools
from typing import List

class MoveType(Enum):
    REGULAR = 0
    EN_PASSANT = 1
    CASTLING = 2
    PAWN_INITIAL_DOUBLE_MOVE = 3
    PAWN_PROMOTE_TO_QUEEN = 4
    PAWN_PROMOTE_TO_KNIGHT = 5
    PAWN_PROMOTE_TO_ROOK = 6
    PAWN_PROMOTE_TO_BISHOP = 7


def read_positions(file_path: str):
    positions = []
    with open(file_path) as file:
        for opening, fen in itertools.zip_longest(*[file]*2):
            positions.append((opening.strip(), fen.strip()))
    return positions

def get_capture_moves(board: chess.Board):
    return [move for move in board.legal_moves if board.is_capture(move)]

def get_captured_piece(board: chess.Board, move: chess.Move):
    captured_piece = board.piece_at(move.to_square)
    if (captured_piece is not None):
        return (move.to_square, captured_piece)
    
    #en passant
    attacking_piece = board.piece_at(move.from_square)
    
    # sanity check that the piece is actually a pawn
    assert attacking_piece is not None and attacking_piece.symbol().lower() == 'p'
    
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
    return (idx, board.piece_at(idx))

def get_moved_piece(board: chess.Board, move: chess.Move):
    return board.piece_at(move.from_square)

piece_type_map = {
    'p': 1,
    'n': 2,
    'b': 3,
    'r': 4,
    'q': 5,
    'k': 6,
    'P': 1,
    'N': 2,
    'B': 3,
    'R': 4,
    'Q': 5,
    'K': 6,
}

def get_piece_type_int(piece: chess.Piece):
    return piece_type_map[piece.symbol()]

def get_move_type(board: chess.Board, move: chess.Move) -> MoveType:
    moved_piece = board.piece_at(move.from_square)
    if (moved_piece.symbol().lower() == 'p'):
        if (move.promotion == chess.QUEEN):
            return MoveType.PAWN_PROMOTE_TO_QUEEN
        if (move.promotion == chess.ROOK):
            return MoveType.PAWN_PROMOTE_TO_ROOK
        if (move.promotion == chess.BISHOP):
            return MoveType.PAWN_PROMOTE_TO_BISHOP
        if (move.promotion == chess.KNIGHT):
            return MoveType.PAWN_PROMOTE_TO_KNIGHT
        if (board.is_capture(move) and board.piece_at(move.to_square) is None):
            return MoveType.EN_PASSANT
        if (abs(chess.square_rank(move.to_square) - chess.square_rank(move.from_square)) == 2):
            return MoveType.PAWN_INITIAL_DOUBLE_MOVE
    if (moved_piece.symbol().lower() == 'k' and abs(move.to_square - move.from_square) == 2):
        return MoveType.CASTLING
    return MoveType.REGULAR

def get_move_value(board: chess.Board, move: chess.Move):
    return (get_move_type(board, move).value << 12) | (move.to_square << 6) | move.from_square

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

# initialize and return a piece count dictionary
def initialize_piece_count(board: chess.Board) -> List[int]:
    piece_count = [0 for _ in range(12)]
    for piece in board.piece_map().values():
        piece_count[get_piece_index(piece)] += 1
    return piece_count

def get_piece_index(piece: chess.Piece):
    return piece_indices[piece.symbol()]
