import chess

from piece_square_tables import evaluate_piece_at_square
from material_info import MaterialInfo, get_material_info
from bitboards import BitboardUtils
from piece import get_piece_value
import util

def get_attacks(board: chess.Board, bitboard_utils: BitboardUtils, color_index: int, ply_from_root: int):
    pieces = bitboard_utils.color_bitboards[color_index]
    attacks = []
    aggregated_attacks = 0
    while (pieces > 0):
        square = bitboard_utils.get_lsb_index(pieces)
        pieces = bitboard_utils.clear_lsb(pieces)
        piece = board.piece_at(square)
        piece_type = util.get_piece_type_int(piece)
        piece_attacks = bitboard_utils.get_piece_attacks(piece_type, color_index, square)
        if (ply_from_root == 0):
            print(piece.symbol())
            bitboard_utils.print_bitboard(piece_attacks)
        attacks.append(piece_attacks)
        aggregated_attacks |= piece_attacks
    return (aggregated_attacks, attacks)

def get_moves(board: chess.Board, bitboard_utils: BitboardUtils, hashed_move: chess.Move, ply_from_root: int, in_q_search: bool = False):

    moves = util.get_capture_moves(board) if (in_q_search) else board.legal_moves

    opponent_material: MaterialInfo = get_material_info(bitboard_utils, chess.BLACK) if (board.ply() % 2 == 0) else get_material_info(bitboard_utils, chess.BLACK)

    # color_index = 0 if (board.ply() % 2 == 0) else 1
    opponent_color_index = 1 if (board.ply() % 2 == 0) else 0
    opponent_attacks, _ = get_attacks(board, bitboard_utils, opponent_color_index, ply_from_root)
    # friendly_attacks, _ = get_attacks(color_index)

    if (ply_from_root == 0):
        print("oppnent attacks")
        bitboard_utils.print_bitboard(opponent_attacks)

    scored_moves = []
    for move in moves:
        score = 0

        if (hashed_move == move):
            scored_moves.append((100000000, move))
            continue

        move_piece = util.get_moved_piece(board, move)
        move_piece_type = util.get_piece_type_int(move_piece)
        move_piece_value = get_piece_value(move_piece_type)
        captured_piece_type = 0
        if (board.is_capture(move)):
            score += 1000000 # capture bias
            _, captured_piece = util.get_captured_piece(board, move)
            captured_piece_type = util.get_piece_type_int(captured_piece)
            captured_piece_value = get_piece_value(captured_piece_type)
            score += 8000000 if (move_piece_value < captured_piece_value) else 2000000
            score += captured_piece_value

            if (opponent_attacks & (1 << move.to_square)):
                score -= move_piece_value


        if (move_piece_type == 1): # Pawn
            move_type = util.get_move_type(board, move)
            is_pawn_promotion = move_type == util.MoveType.PAWN_PROMOTE_TO_KNIGHT \
                or  move_type == util.MoveType.PAWN_PROMOTE_TO_BISHOP \
                    or move_type == util.MoveType.PAWN_PROMOTE_TO_ROOK \
                        or move_type == util.MoveType.PAWN_PROMOTE_TO_QUEEN
            if (is_pawn_promotion and not board.is_capture(move)):
                score += 6000000
        else:
            to_score = evaluate_piece_at_square(move_piece, move.to_square, opponent_material.end_game_percentage())
            from_score = evaluate_piece_at_square(move_piece, move.to_square, opponent_material.end_game_percentage())
            score += to_score - from_score

        scored_moves.append((score, move))

    scored_moves.sort(key=lambda tup: tup[0])
    scored_moves.reverse()
    if (ply_from_root == 0):
        for score,move in scored_moves:
            print(f"move: {move}, score: {score}")
    return [move for _, move in scored_moves]