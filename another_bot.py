import chess
import numpy as np
from math import inf
from agent import Agent
import util
import chess.polyglot

class AnotherChessBot(Agent):
    def __init__(self):
        super().__init__("another agent")

        # Configuration
        self.max_search_time = 0
        self.searching_depth = 0
        self.last_score = 0
        self.push_pop_counter = 0

        # each row is 196 bits or 24 bytes
        # multiply by 10666667 to give us approximately 256 MB transposition table cache
        self.transposition_table = [[0,0,0,0,0] for _ in range(0x800000)]
        
        # Piece-to-history tables per color
        self.history = np.zeros((2, 7, 64), dtype=int)
        
        # Packed evaluation data
        self.packed_data = [
            0x0000000000000000, 0x2328170f2d2a1401, 0x1f1f221929211507, 0x18202a1c2d261507,
            0x252e3022373a230f, 0x585b47456d65321c, 0x8d986f66a5a85f50, 0x0002000300070005,
            0xfffdfffd00060001, 0x2b1f011d20162306, 0x221c0b171f15220d, 0x1b1b131b271c1507,
            0x232d212439321f0b, 0x5b623342826c2812, 0x8db65b45c8c01014, 0x0000000000000000,
            0x615a413e423a382e, 0x6f684f506059413c, 0x82776159705a5543, 0x8b8968657a6a6150,
            0x948c7479826c6361, 0x7e81988f73648160, 0x766f7a7e70585c4e, 0x6c7956116e100000,
            0x3a3d2d2840362f31, 0x3c372a343b3a3838, 0x403e2e343c433934, 0x373e3b2e423b2f37,
            0x383b433c45433634, 0x353d4b4943494b41, 0x46432e354640342b, 0x55560000504f0511,
            0x878f635c8f915856, 0x8a8b5959898e5345, 0x8f9054518f8e514c, 0x96985a539a974a4c,
            0x9a9c67659e9d5f59, 0x989c807a9b9c7a6a, 0xa09f898ba59c6f73, 0xa1a18386a09b7e84,
            0xbcac7774b8c9736a, 0xbab17b7caebd7976, 0xc9ce7376cac57878, 0xe4de6f70dcd87577,
            0xf4ef7175eedc7582, 0xf9fa8383dfe3908e, 0xfffe7a81f4ec707f, 0xdfe79b94e1ee836c,
            0x2027252418003d38, 0x4c42091d31193035, 0x5e560001422c180a, 0x6e6200004d320200,
            0x756c000e5f3c1001, 0x6f6c333f663e3f1d, 0x535b55395c293c1b, 0x2f1e3d5e22005300,
            0x004c0037004b001f, 0x00e000ca00be00ad, 0x02e30266018800eb, 0xffdcffeeffddfff3,
            0xfff9000700010007, 0xffe90003ffeefff4, 0x00000000fff5000d,
        ]

    def get_move(self):
        print("board before")
        print(self.board)
        self.max_search_time = self.timer.remaining_time_nanos() // 4
        self.searching_depth = 1
        self.push_pop_counter = 0

        while (self.searching_depth <= 200 and self.timer.elapsed_time_nanos() < self.max_search_time // 10):
            try:
                # Aspiration windows
                if abs(self.last_score - self.negamax(self.last_score - 20, self.last_score + 20, self.searching_depth)) >= 20:
                    self.negamax(-32000, 32000, self.searching_depth)
                
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

        return self.root_best_move

    def eval_weight(self, item):
        """Extract evaluation weight based on packed data."""
        return int(self.packed_data[item >> 1] >> (item * 32))

    def negamax(self, alpha, beta, depth):
        """Negamax algorithm with alpha-beta pruning."""
        depth = int(depth)
        # Check for timeout
        if self.timer.elapsed_time_nanos() >= self.max_search_time and self.searching_depth > 1:
            raise TimeoutError()

        # Transposition table lookup
        zobrist_key = chess.polyglot.zobrist_hash(self.board)
        tt_index = zobrist_key & 0x7FFFFF
        tt_hash, tt_move_raw, score, tt_depth, tt_bound = self.transposition_table[tt_index]
        
        tt_hit = tt_hash == zobrist_key
        nonPv = alpha + 1 == beta
        inQSearch = depth <= 0

        # Search state
        best_score = int(self.board.ply() - 30000)
        old_alpha = alpha

        move_count = 0 # quietsToCheckTable = [0, 4, 5, 10, 23]
        shift = int(depth * 6)
        if (shift < 0):
            shift = -shift
            quiets_to_check = (388518144 << shift) & 63
        else:
            quiets_to_check = (388518144 >> shift) & 63

        if (tt_hit):
            if (tt_depth >= depth):
                if (tt_bound == 2147483647 and score >= beta):
                    return score
                if (tt_bound == 0 and score <= alpha):
                    return score
                if (nonPv or inQSearch):
                    return score
        elif (depth > 3):
            depth -= 1

        evaluation = int(score if(tt_hit and not inQSearch) else self.eval_board() // 24)

        # Quiescence search
        if (inQSearch):
            best_score = evaluation
            alpha = max(alpha, best_score)
        elif (nonPv and evaluation >= beta and self.try_skip_turn()): # Pruning based on null move observation
            # Reverse Futility Pruning
            # Adaptive Null Move Pruning
            best_score = evaluation - 58 * depth if (depth <= 4) else -self.negamax(-beta, -alpha, int((depth * 100 + beta - evaluation) // 186 - 1))
            self.board.pop()
            self.push_pop_counter -= 1
        if (best_score >= beta):
            return best_score
        if (self.board.is_stalemate()):
            return 0

        moves = util.get_capture_moves(self.board) if (inQSearch) else list(self.board.legal_moves)
        scored_moves = [0 for _ in range(len(moves))]
        tmp = 0
        for move in moves:
            captured_piece_type = 0
            if (self.board.is_capture(move)):
                _, captured_piece = util.get_captured_piece(self.board, move)
                captured_piece_type = util.get_piece_type_int(captured_piece)
            move_piece_type = util.get_piece_type_int(util.get_moved_piece(self.board, move))
            move_raw_value = util.get_move_value(self.board, move)
            score = 1000000 if (tt_hit and move_raw_value == tt_move_raw) else max(
                captured_piece_type * 32768 - move_piece_type - 16384,
                self.history_value(move)
            )
            scored_moves[tmp] = (-score, move)
            tmp += 1

        scored_moves.sort(key=lambda tup: tup[0])

        # Fetch legal moves
        best_move = None
        for _, move in scored_moves:
            if (inQSearch):
                _, captured_piece = util.get_captured_piece(self.board, move)
                capture_piece_type = util.get_piece_type_int(captured_piece)
                shifted_capture_piece_type = ((capture_piece_type * 10) & 0b1_11111_11111)
                partial_table = 0b1_0100110100_1011001110_0110111010_0110000110_0010110100_0000000000
                potential_score = evaluation + (partial_table >> shifted_capture_piece_type)
                if (potential_score <= alpha):
                    break
            
            move_piece_type = util.get_piece_type_int(util.get_moved_piece(self.board, move))
            is_capture_move = self.board.is_capture(move)
            self.bitboard_utils.make_move(move)
            self.board.push(move)
            self.push_pop_counter += 1
            zobrist_key = chess.polyglot.zobrist_hash(self.board)
            self.repetition_table.push(zobrist_key, is_capture_move or move_piece_type == 1) # pawn

            nextDepth = depth if(self.board.is_check()) else depth - 1
            # Late move reduction and history reduction
            reduction = int((depth - nextDepth) * max(
                (move_count * 93 + depth * 144) // 1000 + scored_moves[move_count][0] // 172,
                0
            ))
            if (self.repetition_table.contains(zobrist_key)):
                score = 0
            else:
                while (move_count != 0 
                       and (score := -self.negamax(-alpha-1, -alpha, nextDepth - reduction)) > alpha
                       and reduction != 0):
                    reduction = 0
                if (move_count == 0 or score > alpha):
                    score = -self.negamax(-beta, -alpha, nextDepth)

            self.repetition_table.try_pop()
            self.board.pop()
            self.push_pop_counter -= 1
            self.bitboard_utils.undo_move(move)

            if (score > best_score):
                best_score = score
                alpha = max(alpha, best_score)
                best_move = move
            if (score >= beta):
                if (not self.board.is_capture(move)):
                    tmp = int(evaluation - alpha) >> 31 ^ depth
                    tmp *= tmp
                    for idx in range(move_count):
                        _, malus_move = scored_moves[idx]
                        if (self.board.is_capture(malus_move)):
                            self.increment_history_value(malus_move, -(tmp + tmp * self.history_value(malus_move) / 512))
                    
                    self.increment_history_value(move, tmp - tmp * self.history_value(move) / 512)
                break

            # pruning techniques that break the move loop
            if (nonPv and depth <= 4 and not self.board.is_capture(move)):
                quiets_to_check -= 1
                #Late move pruning
                if (quiets_to_check == 0):
                    break
                elif (evaluation + 127 * depth < alpha): # Futility pruning
                    break

            move_count += 1

        # Update transposition table
        self.transposition_table[tt_index] = (
            chess.polyglot.zobrist_hash(self.board),
            (0 if (best_move is None) else util.get_move_value(self.board, best_move)) if (alpha > old_alpha) else tt_move_raw,
            min(max(best_score, -20000), 20000),
            max(0, depth),
            2147483647 if (best_score >= beta) else alpha - old_alpha)

        self.search_best_move = best_move
        self.lastScore = best_score
        return best_score

    def history_value(self, move: chess.Move):
        """Reference history values."""
        ply = self.board.ply() & 1
        return self.history[ply, util.get_piece_type_int(self.board.piece_at(move.from_square)), move.to_square]
    
    def increment_history_value(self, move: chess.Move, value: int):
        ply = self.board.ply() & 1
        self.history[ply, util.get_piece_type_int(self.board.piece_at(move.from_square)), move.to_square] = self.history_value(move) + value

    def eval_board(self):
        evaluation = 0x000b000a  # Tempo bonus
        pieces = self.bitboard_utils.all_pieces_bitboard
        tmp = 0
        while (pieces != 0):
            square = self.bitboard_utils.get_lsb_index(pieces)
            pieces = self.bitboard_utils.clear_lsb(pieces)
            piece = self.board.piece_at(square)
            piece_type = util.get_piece_type_int(piece)
            piece_is_white = piece.color == chess.WHITE
            king_file = chess.square_file(self.bitboard_utils.white_king_square) if (piece_is_white) else chess.square_file(self.bitboard_utils.black_king_square)
            piece_type -= (square & 0b111 ^ king_file) >> 1 >> piece_type
            square_index = self.eval_weight(112 + piece_type)
            # packed data
            square_index += self.packed_data[piece_type * 64 + square >> 3 ^ (0 if (piece_is_white) else 0b111)] \
                >> (0x01455410 >> square * 4) * 8 & 0xFF00FF
            square_index += self.eval_weight(11 + piece_type) \
                * self.bitboard_utils.get_slider_attacks(min(5, piece_type), square).bit_count()
            # own pawn ahead
            square_index += self.eval_weight(118 + piece_type) \
                * ((0x0101010101010100 << square if(piece_is_white) else 0x0080808080808080 >> 63 - square) \
                    & self.bitboard_utils.get_piece_bitboard(1, piece_is_white)).bit_count()
            is_white_turn = self.board.ply() % 2 == 0
            evaluation += square_index if(piece_is_white == is_white_turn) else -square_index
            tmp += 0x0421100 >> piece_type * 4 & 0xF
        return evaluation * tmp + evaluation // 0x10000 * (24 - tmp)

    def try_skip_turn(self):
        if (self.board.is_check()):
            return False
        self.board.push(chess.Move.null())
        self.push_pop_counter += 1
        return True
