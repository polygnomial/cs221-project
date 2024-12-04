import chess
from dataclasses import dataclass

from bitboards import BitboardUtils
import piece

@dataclass
class MaterialInfo:
    num_pawns: int = 0
    num_knights: int = 0
    num_bishops: int = 0
    num_rooks: int = 0
    num_queens: int = 0
    _internal_score = None
    _internal_end_game_percentage = None

    def num_minor_pieces(self):
        return self.num_knights + self.num_bishops

    def num_major_pieces(self):
        return self.num_rooks + self.num_queens

    def score(self):
        if (self._internal_score is None):
            _internal_score = self.num_pawns * piece.PAWN_VALUE \
                + self.num_knights * piece.KNIGHT_VALUE \
                    + self.num_bishops * piece.BISHOP_VALUE \
                        + self.num_rooks * piece.ROOK_VALUE \
                            + self.num_queens * piece.QUEEN_VALUE
        return _internal_score

    def end_game_percentage(self):
        if (self._internal_end_game_percentage is None):
            queen_endgame_weight: int = 45
            rook_endgame_weight: int = 20
            bishop_endgame_weight: int = 10
            knight_endgame_weight: int = 10

            endgameStartWeight: int = 2 * rook_endgame_weight + 2 * bishop_endgame_weight + 2 * knight_endgame_weight + queen_endgame_weight
            endgame_weight_sum: int = self.num_queens * queen_endgame_weight \
                + self.num_rooks * rook_endgame_weight \
                    + self.num_bishops * bishop_endgame_weight \
                        + self.num_knights * knight_endgame_weight
            self._internal_end_game_percentage = 1 - min(1.0, endgame_weight_sum / float(endgameStartWeight))
        return self._internal_end_game_percentage
    
def get_material_info(bitboard_utils: BitboardUtils, color):
    shifted_color_index: int = (0 if (color == chess.WHITE) else 1) << 3
    material_info: MaterialInfo = MaterialInfo()
    
    material_info.num_pawns = bitboard_utils.piece_bitboards[1 | shifted_color_index].bit_count()
    material_info.num_knights = bitboard_utils.piece_bitboards[2 | shifted_color_index].bit_count()
    material_info.num_bishops = bitboard_utils.piece_bitboards[3 | shifted_color_index].bit_count()
    material_info.num_rooks = bitboard_utils.piece_bitboards[4 | shifted_color_index].bit_count()
    material_info.num_queens = bitboard_utils.piece_bitboards[5 | shifted_color_index].bit_count()
    
    return material_info