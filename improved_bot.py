import chess

from agent import Agent
from transposition import TranspositionEntry, TranspositionTable
from move_ordering import get_moves
from evaluation import eval_board
import util

class ImprovedMiniMaxAgent(Agent):

    def __init__(self):
        super().__init__("improved_negamax_agent")

        # Configuration
        self.max_search_time = None
        self.searching_depth = 0
        self.last_score = 0
        self.push_pop_counter = 0
        self.num_moves = 0

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

        entry: TranspositionEntry = self.transposition_table.get_entry(self.board)
        if (entry.depth >= depth):
            self.search_best_move = entry.move
            return entry.evaluation

        if (in_q_search):
            score = eval_board(self.board, self.bitboard_utils)
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

        moves = get_moves(self.board, self.bitboard_utils, in_q_search)

        next_depth = max(0, depth - 1)
        first_move = True
        for move in moves:

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
            if (alpha + 1 == beta and depth <= 4 and not is_capture_move):
                if (quiet_moves_to_check == 0):
                    break
                quiet_moves_to_check -= 1
        
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
