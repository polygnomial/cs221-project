import chess
import numpy as np
from math import inf
from agent import Agent
from timer import Timer

class AnotherChessBot(Agent):
    def __init__(self, timer: Timer):
        super().__init__("another agent")
        # Configuration
        self.max_search_time = 0
        self.searching_depth = 1
        self.last_score = 0
        self.timer = timer

        # each row is 196 bits or 24 bytes
        # multiply by 10666667 to give us approximately 256 MB transposition table cache
        self.transposition_table = np.zeros(10666667, dtype='u8, u2, i4, i4, i4')
        
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
        self.max_search_time = self.timer.remaining_time_nanos() // 4
        self.searching_depth = 1

        while (self.searching_depth <= 200 and self.timer.elapsed_time_nanos() < self.max_search_time // 10):
            try:
                # Aspiration windows
                if abs(self.last_score - self.negamax(self.last_score - 20, self.last_score + 20, self.searching_depth)) >= 20:
                    self.negamax(-32000, 32000, self.searching_depth)
                self.root_best_move = self.search_best_move
            except TimeoutError:
                break

            self.searching_depth += 1

        return self.root_best_move

    def eval_weight(self, item):
        """Extract evaluation weight based on packed data."""
        return (self.packed_data[item >> 1] >> (item * 32)) & 0xFFFFFFFF

    def negamax(self, alpha, beta, depth):
        """Negamax algorithm with alpha-beta pruning."""
        # Check for timeout
        if self.timer.elapsed_time_nanos() >= self.max_search_time and self.searching_depth > 1:
            raise TimeoutError()

        # Transposition table lookup
        zobrist_key = chess.polyglot.zobrist_hash(self.board)
        tt_index = zobrist_key & 0x7FFFFF
        tt_hit, tt_move_raw, score, tt_depth, tt_bound = self.transposition_table[tt_index]
        tt_hit = tt_hit == zobrist_key

        # Search state
        eval = 11  # Tempo bonus
        best_score = -inf
        old_alpha = alpha

        # Quiescence search
        if depth <= 0:
            alpha = max(alpha, best_score)
            return best_score

        # Prune with null move heuristic if applicable
        if alpha >= beta:
            return alpha

        # Fetch legal moves
        for move in self.board.legal_moves:
            # Ordering moves (hash, captures, history)
            if tt_hit and move.raw_value == tt_move_raw:
                continue

            self.board.make_move(move)
            score = -self.negamax(-beta, -alpha, depth - 1)
            self.board.undo_move()

            if score > best_score:
                best_score = score
                alpha = max(alpha, score)
                self.search_best_move = move

            if alpha >= beta:
                break

        # Update transposition table
        self.transposition_table[tt_index] = (
            zobrist_key,
            best_move.raw if (alpha > old_alpha) else tt_move_raw,
            min(max(best_score, -20000), 20000),
            max(0, depth),
            2147483647 if (best_score >= beta) else alpha - old_alpha)

        self.lastScore = best_score
        return best_score

    def history_value(self, move):
        """Reference history values."""
        ply = self.board.ply_count % 2
        return self.history[ply, move.piece_type, move.target_square]

# Add supporting methods for handling board operations, timing, zobrist hashing, etc.
