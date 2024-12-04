import chess
from dataclasses import dataclass

from bits import BITS
from bitboards import BitboardUtils
from piece_square_tables import evaluate_piece_at_square
from material_info import MaterialInfo, get_material_info
from precomuted_eval_data import PRECOMPUTED_EVAL_DATA
import util

@dataclass
class EvaluationData:
    material_score: int = 0
    mop_up_score: int = 0
    piece_square_score: int = 0
    pawn_score: int = 0
    pawn_shield_score: int = 0

    def sum(self):
        return self.material_score + self.mop_up_score + self.piece_square_score + self.pawn_score + self.pawn_shield_score

def evaluate_piece_square_tables(board: chess.Board, bitboard_utils: BitboardUtils, color: chess.Color, end_game_percentage: float):
    value: int = 0
    color_index: int = 0 if (color == chess.WHITE) else 1
    pieces = bitboard_utils.color_bitboards[color_index]
    while (pieces > 0):
        square = bitboard_utils.get_lsb_index(pieces)
        pieces = bitboard_utils.clear_lsb(pieces)
        piece = board.piece_at(square)

        value += evaluate_piece_at_square(piece=piece, square=square, end_game_percentage=end_game_percentage)

    return value

# As game transitions to endgame, and if up material, then encourage moving king closer to opponent king
def evaluate_mop_up(bitboard_utils: BitboardUtils, color: chess.Color, friendly_material: MaterialInfo, opponent_material: MaterialInfo):
    if (friendly_material.score() <= opponent_material.score() + 200 or opponent_material.end_game_percentage() <= 0):
        return 0

    mopUpScore: int = 0

    friendly_king_square = bitboard_utils.white_king_square if (color == chess.WHITE) else bitboard_utils.black_king_square
    opponent_king_square = bitboard_utils.black_king_square if (color == chess.WHITE) else bitboard_utils.white_king_square
    
    # Encourage moving king closer to opponent king
    mopUpScore += (14 - util.manhattan_distance(friendly_king_square, opponent_king_square)) * 4

    # Encourage pushing opponent king to edge of board
    mopUpScore += util.manhattan_distance_from_center(opponent_king_square) * 10
    return int(mopUpScore * opponent_material.end_game_percentage())

def evaluate_pawns(bitboard_utils: BitboardUtils, color: chess.Color):
    isolated_pawn_penalty_by_count = [0, -10, -25, -50, -75, -75, -75, -75, -75]
    passed_pawn_bonuses = [0, 120, 80, 50, 30, 15, 15]

    shifted_color_index: int = (0 if (color == chess.WHITE) else 1) << 3
    shifted_opponent_color_index: int = (1 if (color == chess.WHITE) else 0) << 3

    pawns = bitboard_utils.piece_bitboards[1 | shifted_color_index]
    opponent_pawns = bitboard_utils.piece_bitboards[1 | shifted_opponent_color_index]
    friendly_pawns = bitboard_utils.piece_bitboards[1 | shifted_color_index]
    masks = BITS.white_passed_pawn_mask if (color == chess.WHITE) else BITS.black_passed_pawn_mask
    bonus: int = 0
    num_isolated_pawns: int = 0

    while (pawns > 0):
        square = bitboard_utils.get_lsb_index(pawns)
        pawns = bitboard_utils.clear_lsb(pawns)
        passed_mask = masks[square]
        
        # Is passed pawn
        if ((opponent_pawns & passed_mask) == 0):
            rank: int = chess.square_rank(square)
            num_squares_from_promotion: int = (7 - rank) if (color == chess.WHITE) else rank
            bonus += passed_pawn_bonuses[num_squares_from_promotion]

        # Is isolated pawn
        if ((friendly_pawns & BITS.adjacent_file_masks[chess.square_file(square)]) == 0):
            num_isolated_pawns += 1

    return bonus + isolated_pawn_penalty_by_count[num_isolated_pawns]

def is_friendly_pawn_at_square(board: chess.Board, square: int, color: chess.Color):
    piece = board.piece_at(square)
    if (piece is None):
        return False
    piece_type = util.get_piece_type_int(piece)
    return piece.color == color and piece_type == 1

def evaluate_king_pawn_shield(board: chess.Board, bitboard_utils: BitboardUtils, color: chess.Color, opponent_material: MaterialInfo, enemy_piece_square_score: float):
    king_pawn_shield_scores = [4, 7, 4, 3, 6, 3]
    if (opponent_material.end_game_percentage() >= 1):
        return 0

    penalty: int = 0

    color_index: int = 0 if (color == chess.WHITE) else 1
    opponent_color_index: int = 1 if (color == chess.WHITE) else 0
    shifted_color_index = color_index << 3
    shifted_opponent_color_index = opponent_color_index << 3
    king_square = bitboard_utils.white_king_square if(color == chess.WHITE) else bitboard_utils.black_king_square
    king_file = chess.square_file(king_square)

    uncastled_king_penalty = 0

    if (king_file <= 2 or king_file >= 5):
        squares = PRECOMPUTED_EVAL_DATA.pawn_shield_squares_white[king_square] if (color == chess.WHITE) \
            else PRECOMPUTED_EVAL_DATA.pawn_shield_squares_black[king_square]

        for i in range(len(squares)//2):
            if (not is_friendly_pawn_at_square(board=board, square=squares[i], color=color)):
                if (len(squares) > 3 and is_friendly_pawn_at_square(board=board, square=squares[i + 3], color=color)):
                    penalty += king_pawn_shield_scores[i + 3]
                else:
                    penalty += king_pawn_shield_scores[i]

        penalty *= penalty
    else:
        enemy_development_score = util.clamp((enemy_piece_square_score + 10) / 130.0, 0, 1)
        uncastled_king_penalty = int(50 * enemy_development_score)

    open_file_against_king_penalty: int = 0

    if (opponent_material.num_rooks > 1 or (opponent_material.num_rooks > 0 and opponent_material.num_queens > 0)):
        clamped_king_file = util.clamp(king_file, 1, 6)
        friendly_pawns = bitboard_utils.piece_bitboards[1 | shifted_color_index]
        opponent_pawns = bitboard_utils.piece_bitboards[1 | shifted_opponent_color_index]
        for attack_file in range(clamped_king_file, clamped_king_file + 2):
            fileMask = BITS.file_mask[attack_file]
            is_king_file = attack_file == king_file
            if ((opponent_pawns & fileMask) == 0):
                open_file_against_king_penalty += 25 if (is_king_file) else 15
                if ((friendly_pawns & fileMask) == 0):
                    open_file_against_king_penalty += 15 if (is_king_file) else 10

    pawn_shield_weight = 1 - opponent_material.end_game_percentage()
    
    # if the opponent does not have a queen, pawn shielding matters less
    if (bitboard_utils.piece_bitboards[5 | shifted_color_index].bit_count() == 0):
        pawn_shield_weight *= 0.6

    return int((-penalty - uncastled_king_penalty - open_file_against_king_penalty) * pawn_shield_weight)

def eval_board(board: chess.Board, bitboard_utils: BitboardUtils):
    white_evaluation = EvaluationData()
    black_evaluation = EvaluationData()

    white_material: MaterialInfo = get_material_info(bitboard_utils, chess.WHITE)
    black_material: MaterialInfo = get_material_info(bitboard_utils, chess.BLACK)

    # Score based on number (and type) of pieces on board
    white_evaluation.material_score = white_material.score()
    black_evaluation.material_score = black_material.score()
    
    # Score based on positions of pieces
    white_evaluation.piece_square_score = evaluate_piece_square_tables(board, bitboard_utils, chess.WHITE, black_material.end_game_percentage())
    black_evaluation.piece_square_score = evaluate_piece_square_tables(board, bitboard_utils, chess.BLACK, white_material.end_game_percentage())
    
    # Encourage using own king to push enemy king to edge of board in winning endgame
    white_evaluation.mop_up_score = evaluate_mop_up(bitboard_utils, chess.WHITE, white_material, black_material)
    black_evaluation.mop_up_score = evaluate_mop_up(bitboard_utils, chess.BLACK, black_material, white_material)

    white_evaluation.pawn_score = evaluate_pawns(bitboard_utils, chess.WHITE)
    black_evaluation.pawn_score = evaluate_pawns(bitboard_utils, chess.BLACK)

    white_evaluation.pawn_shield_score = evaluate_king_pawn_shield(board, bitboard_utils, chess.WHITE, black_material, black_evaluation.piece_square_score)
    black_evaluation.pawn_shield_score = evaluate_king_pawn_shield(board, bitboard_utils, chess.BLACK, white_material, white_evaluation.piece_square_score)

    perspective = 1 if (board.turn == chess.WHITE) else -1
    return (white_evaluation.sum() - black_evaluation.sum()) * perspective