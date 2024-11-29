import chess
import chess.polyglot
from dataclasses import dataclass
# from typing import List

from agent import Agent
from bits import Bits
from piece_square_tables import evaluate_piece_at_square
from precomuted_eval_data import PrecomputedEvaluationData
from transposition import TranspositionEntry, TranspositionTable
import util


PAWN_VALUE = 100
KNIGHT_VALUE = 200
BISHOP_VALUE = 320
ROOK_VALUE = 500
QUEEN_VALUE = 900

@dataclass
class EvaluationData:
    material_score: int = 0
    mop_up_score: int = 0
    piece_square_score: int = 0
    pawn_score: int = 0
    pawn_shield_score: int = 0

    def sum(self):
        return self.material_score + self.mop_up_score + self.piece_square_score + self.pawn_score + self.pawn_shield_score
    

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
            self._internal_score = self.num_pawns * PAWN_VALUE \
                + self.num_knights * KNIGHT_VALUE \
                    + self.num_bishops * BISHOP_VALUE \
                        + self.num_rooks * ROOK_VALUE \
                            + self.num_queens * QUEEN_VALUE
        return self._internal_score

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

class ImprovedMiniMaxAgent(Agent):

    def __init__(self):
        super().__init__("improved_negamax_agent")

        # Configuration
        self.max_search_time = None
        self.searching_depth = 0
        self.last_score = 0
        self.push_pop_counter = 0
        self.num_moves = 0

        self.precomputed_evaluation_data = PrecomputedEvaluationData()
        self.bits = Bits()
        self.transposition_table = TranspositionTable()

    
    def negamax(self, alpha: float, beta: float, depth: int):
        # Check for timeout
        if self.timer.elapsed_time_nanos() >= self.max_search_time and self.searching_depth > 1:
            raise TimeoutError()

        if (self.board.is_stalemate() or self.board.is_insufficient_material()):
            return 0
        if (self.board.is_checkmate()):
            return float('inf') if (self.board.outcome().winner == self.color) else float('-inf')
        
        in_q_search: bool = depth <= 0
        best_score = float('-inf')
        best_move = None

        quiet_moves_to_check = (0b_010111_001010_000101_000100_000000 >> depth * 6) & 0b111111

        zobrist_hash = chess.polyglot.zobrist_hash(self.board)
        entry: TranspositionEntry = self.transposition_table.get_entry(self.board)
        if (entry.key == zobrist_hash and entry.depth >= depth):
            self.search_best_move = entry.move
            return entry.evaluation

        if (in_q_search):
            score = self.eval_board()
            if (score >= beta):
                return score
            if (score > alpha):
                alpha = score

        # moves = util.get_capture_moves(self.board) if (in_q_search) else self.board.legal_moves

        # scored_moves = []
        # for move in moves:
        #     move_piece_type = util.get_piece_type_int(util.get_moved_piece(self.board, move))
        #     captured_piece_type = 0
        #     if (self.board.is_capture(move)):
        #         _, captured_piece = util.get_captured_piece(self.board, move)
        #         captured_piece_type = util.get_piece_type_int(captured_piece)

        #     # technically the score is captured piece - moved piece but we are using default sorting
        #     # so we want to flip the sign in order to make sure the best scores come first
        #     scored_moves.append((move_piece_type * 16384 - captured_piece_type * 32768, move))

        # scored_moves.sort(key=lambda tup: tup[0])

        scored_moves = self.get_moves(in_q_search)

        next_depth = max(0, depth - 1)
        first_move = True
        for _, move in scored_moves:

            move_piece_type = util.get_piece_type_int(util.get_moved_piece(self.board, move))
            is_capture_move = self.board.is_capture(move)

            # make the move
            self.bitboard_utils.make_move(move)
            self.board.push(move)
            self.push_pop_counter += 1

            # recursive call delegating to the other player using principal variation search
            if (self.board.is_repetition()):
                score = 0
            elif (first_move or in_q_search):
                score = -self.negamax(alpha=-beta, beta=-alpha, depth=next_depth)
            else:
                score = -self.negamax(alpha=-alpha-1, beta=-alpha, depth=next_depth)
                if ( score > alpha and beta - alpha > 1):
                    score = -self.negamax(alpha=-beta, beta=-alpha, depth=next_depth)
            
            # reset board
            self.board.pop()
            self.bitboard_utils.undo_move(move)
            self.push_pop_counter -= 1

            if (score > best_score):
                best_score = score
                best_move = move

                alpha = max(alpha, best_score)

            if (score >= beta):
                break

            # pruning techniques that break the move loop
            quiet_moves_to_check -= 1
            if (alpha + 1 == beta and depth <= 4 and not is_capture_move and quiet_moves_to_check == 0):
                break
        
        if (best_move is None):
            return alpha
        self.search_best_move = best_move
        self.transposition_table.store_evaluation(board=self.board, depth=depth, evaluation=best_score, move=best_move)
        return best_score

    def get_move(self):
        if (self.max_search_time is None):
            self.max_search_time = self.timer.remaining_time_nanos() // 70
        if (self.num_moves > 70):
            self.max_search_time = self.timer.remaining_time_nanos() // 4
        self.searching_depth = 1
        self.push_pop_counter = 0
        self.color = chess.WHITE if (self.board.ply() % 2 == 0) else chess.BLACK

        while (self.searching_depth <= 10 and self.timer.elapsed_time_nanos() < self.max_search_time):
            try:
                self.negamax(alpha=float('-inf'), beta=float('inf'), depth=self.searching_depth)
                
                self.root_best_move = self.search_best_move
            except TimeoutError:
                print("timeout exception")
                break

            self.searching_depth += 1
        
        # needed to fix board state after a timeout
        while(self.push_pop_counter > 0):
            move = self.board.pop()
            # only undo null moves
            if (bool(move)):
                self.bitboard_utils.undo_move(move)
            self.push_pop_counter -= 1

        print("board is:")
        print(self.board)
        print("The best move is:")
        print(self.root_best_move)
        self.num_moves += 1
        return self.root_best_move

    def get_material_info(self, color):
        shifted_color_index: int = (0 if (color == chess.WHITE) else 1) << 3
        material_info: MaterialInfo = MaterialInfo()
        
        material_info.num_pawns = self.bitboard_utils.piece_bitboards[1 | shifted_color_index].bit_count()
        material_info.num_knights = self.bitboard_utils.piece_bitboards[2 | shifted_color_index].bit_count()
        material_info.num_bishops = self.bitboard_utils.piece_bitboards[3 | shifted_color_index].bit_count()
        material_info.num_rooks = self.bitboard_utils.piece_bitboards[4 | shifted_color_index].bit_count()
        material_info.num_queens = self.bitboard_utils.piece_bitboards[5 | shifted_color_index].bit_count()
        
        return material_info

    def evaluate_piece_square_tables(self, color: chess.Color, end_game_percentage: float):
        value: int = 0
        color_index: int = 0 if (color == chess.WHITE) else 1
        pieces = self.bitboard_utils.color_bitboards[color_index]
        while (pieces > 0):
            square = self.bitboard_utils.get_lsb_index(pieces)
            pieces = self.bitboard_utils.clear_lsb(pieces)
            piece = self.board.piece_at(square)

            value += evaluate_piece_at_square(piece=piece, square=square, end_game_percentage=end_game_percentage)

        return value

    # // As game transitions to endgame, and if up material, then encourage moving king closer to opponent king
    def evaluate_mop_up(self, color: chess.Color, friendly_material: MaterialInfo, opponent_material: MaterialInfo):
        if (friendly_material.score() <= opponent_material.score() + 200 or opponent_material.end_game_percentage() <= 0):
            return 0

        mopUpScore: int = 0

        friendly_king_square = self.bitboard_utils.white_king_square if (color == chess.WHITE) else self.bitboard_utils.black_king_square
        opponent_king_square = self.bitboard_utils.black_king_square if (color == chess.WHITE) else self.bitboard_utils.white_king_square
        
        # Encourage moving king closer to opponent king
        mopUpScore += (14 - util.manhattan_distance(friendly_king_square, opponent_king_square)) * 4

        # Encourage pushing opponent king to edge of board
        mopUpScore += util.manhattan_distance_from_center(opponent_king_square) * 10
        return int(mopUpScore * opponent_material.end_game_percentage())

    def evaluate_pawns(self, color: chess.Color):
        isolated_pawn_penalty_by_count = [0, -10, -25, -50, -75, -75, -75, -75, -75]
        passed_pawn_bonuses = [0, 120, 80, 50, 30, 15, 15]

        shifted_color_index: int = (0 if (color == chess.WHITE) else 1) << 3
        shifted_opponent_color_index: int = (1 if (color == chess.WHITE) else 0) << 3

        pawns = self.bitboard_utils.piece_bitboards[1 | shifted_color_index]
        opponent_pawns = self.bitboard_utils.piece_bitboards[1 | shifted_opponent_color_index]
        friendly_pawns = self.bitboard_utils.piece_bitboards[1 | shifted_color_index]
        masks = self.bits.white_passed_pawn_mask if (color == chess.WHITE) else self.bits.black_passed_pawn_mask
        bonus: int = 0
        num_isolated_pawns: int = 0

        while (pawns > 0):
            square = self.bitboard_utils.get_lsb_index(pawns)
            pawns = self.bitboard_utils.clear_lsb(pawns)
            passed_mask = masks[square]
            
            # Is passed pawn
            if ((opponent_pawns & passed_mask) == 0):
                rank: int = chess.square_rank(square)
                num_squares_from_promotion: int = (7 - rank) if (color == chess.WHITE) else rank
                bonus += passed_pawn_bonuses[num_squares_from_promotion]

            # Is isolated pawn
            if ((friendly_pawns & self.bits.adjacent_file_masks[chess.square_file(square)]) == 0):
                num_isolated_pawns += 1

        return bonus + isolated_pawn_penalty_by_count[num_isolated_pawns]

    def is_friendly_pawn_at_square(self, square: int, color: chess.Color):
        piece = self.board.piece_at(square)
        if (piece is None):
            return False
        piece_type = util.get_piece_type_int(piece)
        return piece.color == color and piece_type == 1

    def evaluate_king_pawn_shield(self, color: chess.Color, opponent_material: MaterialInfo, enemy_piece_square_score: float):
        king_pawn_shield_scores = [4, 7, 4, 3, 6, 3]
        if (opponent_material.end_game_percentage() >= 1):
            return 0

        penalty: int = 0

        color_index: int = 0 if (color == chess.WHITE) else 1
        opponent_color_index: int = 1 if (color == chess.WHITE) else 0
        shifted_color_index = color_index << 3
        shifted_opponent_color_index = opponent_color_index << 3
        king_square = self.bitboard_utils.white_king_square if(color == chess.WHITE) else self.bitboard_utils.black_king_square
        king_file = chess.square_file(king_square)

        uncastled_king_penalty = 0

        if (king_file <= 2 or king_file >= 5):
            squares = self.precomputed_evaluation_data.pawn_shield_squares_white[king_square] if (color == chess.WHITE) \
                else self.precomputed_evaluation_data.pawn_shield_squares_black[king_square]

            for i in range(len(squares)//2):
                if (not self.is_friendly_pawn_at_square(square=squares[i], color=color)):
                    if (len(squares) > 3 and self.is_friendly_pawn_at_square(square=squares[i + 3], color=color)):
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
            friendly_pawns = self.bitboard_utils.piece_bitboards[1 | shifted_color_index]
            opponent_pawns = self.bitboard_utils.piece_bitboards[1 | shifted_opponent_color_index]
            for attack_file in range(clamped_king_file, clamped_king_file + 2):
                fileMask = self.bits.file_mask[attack_file]
                is_king_file = attack_file == king_file
                if ((opponent_pawns & fileMask) == 0):
                    open_file_against_king_penalty += 25 if (is_king_file) else 15
                    if ((friendly_pawns & fileMask) == 0):
                        open_file_against_king_penalty += 15 if (is_king_file) else 10

        pawn_shield_weight = 1 - opponent_material.end_game_percentage()
        
        # if the opponent does not have a queen, pawn shielding matters less
        if (self.bitboard_utils.piece_bitboards[5 | shifted_color_index].bit_count() == 0):
            pawn_shield_weight *= 0.6

        return int((-penalty - uncastled_king_penalty - open_file_against_king_penalty) * pawn_shield_weight)

    def eval_board(self):
        white_evaluation = EvaluationData()
        black_evaluation = EvaluationData()

        white_material: MaterialInfo = self.get_material_info(chess.WHITE)
        black_material: MaterialInfo = self.get_material_info(chess.BLACK)

        # Score based on number (and type) of pieces on board
        white_evaluation.material_score = white_material.score()
        black_evaluation.material_score = black_material.score()
        
        # Score based on positions of pieces
        white_evaluation.piece_square_score = self.evaluate_piece_square_tables(chess.WHITE, black_material.end_game_percentage())
        black_evaluation.piece_square_score = self.evaluate_piece_square_tables(chess.BLACK, white_material.end_game_percentage())
        
        # Encourage using own king to push enemy king to edge of board in winning endgame
        white_evaluation.mop_up_score = self.evaluate_mop_up(chess.WHITE, white_material, black_material)
        black_evaluation.mop_up_score = self.evaluate_mop_up(chess.BLACK, black_material, white_material)

        white_evaluation.pawn_score = self.evaluate_pawns(chess.WHITE)
        black_evaluation.pawn_score = self.evaluate_pawns(chess.BLACK)

        white_evaluation.pawn_shield_score = self.evaluate_king_pawn_shield(chess.WHITE, black_material, black_evaluation.piece_square_score)
        black_evaluation.pawn_shield_score = self.evaluate_king_pawn_shield(chess.BLACK, white_material, white_evaluation.piece_square_score)

        perspective = 1 if (self.board.turn == chess.WHITE) else -1
        return (white_evaluation.sum() - black_evaluation.sum()) * perspective
    
    def get_piece_value(self, piece_type):
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
    
    def get_attacks(self, color_index: int):
        pieces = self.bitboard_utils.color_bitboards[color_index]
        attacks = []
        aggregated_attacks = 0
        while (pieces > 0):
            square = self.bitboard_utils.get_lsb_index(pieces)
            pieces = self.bitboard_utils.clear_lsb(pieces)
            piece = self.board.piece_at(square)
            piece_type = util.get_piece_type_int(piece)
            board = self.bitboard_utils.get_piece_attacks(piece_type, color_index, square)
            attacks.append(board)
            aggregated_attacks |= board
        return (aggregated_attacks, attacks)
    
    def get_moves(self, in_q_search: bool):

        moves = util.get_capture_moves(self.board) if (in_q_search) else self.board.legal_moves

        opponent_material: MaterialInfo = self.get_material_info(chess.BLACK) if (self.board.ply() % 2 == 0) else self.get_material_info(chess.BLACK)

        # color_index = 0 if (self.board.ply() % 2 == 0) else 1
        opponent_color_index = 1 if (self.board.ply() % 2 == 0) else 0
        opponent_attacks, _ = self.get_attacks(opponent_color_index)
        # friendly_attacks, _ = self.get_attacks(color_index)

        scored_moves = []
        for move in moves:
            score = 0

            move_piece = util.get_moved_piece(self.board, move)
            move_piece_type = util.get_piece_type_int(move_piece)
            move_piece_value = self.get_piece_value(move_piece_type)
            captured_piece_type = 0
            if (self.board.is_capture(move)):
                score += 1000000 # capture bias
                _, captured_piece = util.get_captured_piece(self.board, move)
                captured_piece_type = util.get_piece_type_int(captured_piece)
                captured_piece_value = self.get_piece_value(captured_piece_type)
                score += 8000000 if (move_piece_value < captured_piece_value) else 2000000
                score += captured_piece_value

                if (opponent_attacks & (1 << move.to_square)):
                    score -= move_piece_value


            if (move_piece_type == 1): # Pawn
                move_type = util.get_move_type(self.board, move)
                is_pawn_promotion = move_type == util.MoveType.PAWN_PROMOTE_TO_KNIGHT \
                    or  move_type == util.MoveType.PAWN_PROMOTE_TO_BISHOP \
                        or move_type == util.MoveType.PAWN_PROMOTE_TO_ROOK \
                            or move_type == util.MoveType.PAWN_PROMOTE_TO_QUEEN
                if (is_pawn_promotion and not self.board.is_capture(move)):
                    score += 6000000
            else:
                to_score = evaluate_piece_at_square(move_piece, move.to_square, opponent_material.end_game_percentage())
                from_score = evaluate_piece_at_square(move_piece, move.to_square, opponent_material.end_game_percentage())
                score += to_score - from_score

            scored_moves.append((score, move))

        scored_moves.sort(key=lambda tup: tup[0])
        scored_moves.reverse()
        return scored_moves