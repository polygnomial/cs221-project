import chess
from typing import Optional

from agent import Agent
from transposition import TranspositionBound, TranspositionEntry, TranspositionTable
from move_ordering import get_moves
from evaluation import eval_board
import util

class YetAnotherAgent(Agent):
    def __init__(self, debug: bool = False):
        super().__init__("YetAnotherAgent")

        # Configuration
        self.max_search_time = None
        self.searching_depth = 0
        self.last_score = 0
        self.push_pop_counter = 0
        self.num_moves = 0
        self.search_best_move = None
        self.root_best_move = None
        self.transposition_table = TranspositionTable()
        self.debug = debug
        self.captured_piece = None

    def get_entry_score(self, entry: TranspositionEntry, alpha: float, beta: float):
        if (entry.bound == TranspositionBound.EXACT):
            return entry.evaluation
        if (entry.bound == TranspositionBound.UPPER and entry.evaluation <= alpha):
            return entry.evaluation
        if (entry.bound == TranspositionBound.LOWER and entry.evaluation >= beta):
            return entry.evaluation
        return None

    def make_move(self, move: chess.Move):
        self.captured_piece = None
        if self.board.is_capture(move):
            self.captured_piece = util.get_captured_piece(self.board, move)[1]
            self.piece_count[util.piece_indices[self.captured_piece.symbol()]] -= 1

        self.bitboard_utils.make_move(move)
        self.board.push(move)
        self.push_pop_counter += 1

    def undo_move(self, move: chess.Move):
            self.board.pop()
            self.bitboard_utils.undo_move(move)
            self.push_pop_counter -= 1

            if self.captured_piece is not None:
                self.piece_count[util.piece_indices[self.captured_piece.symbol()]] += 1

    def quiesce(self, alpha: float, beta: float):
        stand_pat = eval_board(self.board, self.bitboard_utils)
        if(stand_pat >= beta):
            return (beta, None)
        alpha = max(alpha, stand_pat)
        
        for move in self.board.legal_moves:
            if self.board.is_capture(move):
                self.make_move(move)
                score, _ = self.quiesce(alpha=-beta, beta=-alpha)
                score = -score
                self.undo_move(move)

                if(score >= beta):
                    return (beta, None)
                alpha = max(score, alpha)
        
        return (alpha, None)

    def pv_search(self, alpha: float, beta: float, depth: int, ply_from_root: int, non_pv: bool):
        if(depth == 0):
            return self.quiesce(alpha, beta)
        
        best_score = None
        best_move = None

        moves = get_moves(board=self.board, bitboard_utils=self.bitboard_utils, hashed_move=None, ply_from_root=ply_from_root)

        if (len(moves) == 0):
            print("len moves is 0")
            print(self.board)
            if (self.board.is_stalemate() or self.board.is_insufficient_material()):
                return (0, None)
            if (self.board.is_checkmate()):
                return (float('inf') if (self.board.outcome().winner == chess.WHITE) else float('-inf'), None)

        for move in moves:
            self.make_move(move)
            if (best_score is None): # full search on first move
                response = self.pv_search(alpha=-beta, beta=-alpha, depth=depth - 1, ply_from_root=ply_from_root+1, non_pv=non_pv)
                if (response[0] is None):
                    print("BEST_SCORE")
                    print(f"alpha={-beta}, beta={-alpha}")
                best_score = -response[0]
                best_move = move
                if (ply_from_root == 0):
                    print(move)
                    print(self.board)
                    print(best_score)
            else:
                response = self.pv_search(alpha=-alpha-1, beta=-alpha, depth=depth - 1, ply_from_root=ply_from_root+1, non_pv=True)
                if (response[0] is None):
                    print("non pv")
                    print(f"alpha={-beta}, beta={-alpha}")
                score = -response[0]
                if (score > alpha and beta - alpha > 1):
                    response = self.pv_search(alpha=-beta, beta=-alpha, depth=depth - 1, ply_from_root=ply_from_root+1, non_pv=False)
                if (response[0] is None):
                    print("non pv second")
                    print(f"alpha={-beta}, beta={-alpha}")
                    score = -response[0]
                if (best_score < score):
                    best_score = score
                    best_move = move
                if (ply_from_root == 0):
                    print(move)
                    print(self.board)
                    print(score)
            self.undo_move(move)

            if (best_score >= beta):
                return (best_score, None)
            
            alpha = max(best_score, alpha)
            
            alpha = max(alpha, best_score)
        
        return (best_score, best_move)

    def get_move(self):
        _, move = self.pv_search(alpha=float('-inf'), beta=float('inf'), depth=4, ply_from_root=0, non_pv=False)
        return move
    

# class YetAnotherAgent(Agent):
#     def __init__(self, debug: bool = False):
#         super().__init__("YetAnotherAgent")

#         # Configuration
#         self.max_search_time = None
#         self.searching_depth = 0
#         self.last_score = 0
#         self.push_pop_counter = 0
#         self.num_moves = 0
#         self.search_best_move = None
#         self.root_best_move = None
#         self.transposition_table = TranspositionTable()
#         self.debug = debug

#     def get_entry_score(self, entry: TranspositionEntry, alpha: float, beta: float):
#         if (entry.bound == TranspositionBound.EXACT):
#             return entry.evaluation
#         if (entry.bound == TranspositionBound.UPPER and entry.evaluation <= alpha):
#             return entry.evaluation
#         if (entry.bound == TranspositionBound.LOWER and entry.evaluation >= beta):
#             return entry.evaluation
#         return None

#     def negamax(self, alpha: float, beta: float, depth: int, ply_from_root: int):
#         if self.timer.elapsed_time_nanos() >= self.max_search_time and self.searching_depth > 1:
#             raise TimeoutError()
#         if (self.board.is_stalemate() or self.board.is_insufficient_material()):
#             return 0
#         if (self.board.is_checkmate()):
#             return float('inf') if (self.board.outcome().winner == chess.WHITE) else float('-inf')
        
#         in_q_search = depth <= 0
#         non_pv: bool = alpha + 1 == beta

#         entry: TranspositionEntry = self.transposition_table.get_entry(self.board)
#         if (entry is not None and entry.depth >= depth and (ply_from_root == 0 or non_pv or in_q_search)):
#             score = self.get_entry_score(entry, alpha, beta)
#             if (score is not None):
#                 if (ply_from_root == 0):
#                     self.search_best_move = entry.move
#                 return score
        
#         prev_best_move = None
#         if(ply_from_root == 0):
#             prev_best_move = self.root_best_move
#         elif (entry is not None):
#             prev_best_move = entry.move

        
#         if (in_q_search):
#             evaluation = eval_board(self.board, self.bitboard_utils)
#             if (evaluation >= beta):
#                 return beta
#             alpha = max(alpha, evaluation)
#         # elif (non_pv):
#         #     evaluation = eval_board(self.board, self.bitboard_utils)
#         #     if (evaluation >= beta and depth <= 4):
#         #         self.board.push(chess.Move.null())
#         #         self.push_pop_counter += 1
#         #         # TODO maybe remove
#         #         score = -self.negamax(alpha=-beta, beta=-alpha, depth=(depth * 100 + beta - evaluation) // 186 - 1, ply_from_root=ply_from_root+1)
#         #         self.board.pop()
#         #         self.push_pop_counter -= 1


#         quiets_to_check = 0b_010111_001010_000101_000100_000000 >> max(0, depth) * 6 & 0b111111

#         moves = get_moves(self.board, self.bitboard_utils, prev_best_move, in_q_search)

#         best_move = None
#         evaluation_bound = TranspositionBound.UPPER

#         if (self.debug and ply_from_root == 0):
#             print("====================")
#             print(self.board)
#         move_count = 0
#         for move in moves:
#             # if the move is a capture, decrement the count of the captured piece
#             captured_piece = None
#             if self.board.is_capture(move):
#                 _, captured_piece = util.get_captured_piece(self.board, move)
#                 self.piece_count[util.piece_indices[captured_piece.symbol()]] -= 1

#             self.bitboard_utils.make_move(move)
#             self.board.push(move)
#             self.push_pop_counter += 1

#             if (self.debug and ply_from_root == 0):
#                 print(move)
#                 print(self.board)

#             if (not in_q_search and self.board.is_repetition(2)):
#                 score = 0
#             else:
#                 needs_full_search = True
#                 if (depth >= 3 and move_count >= 3 and captured_piece is None):
#                     score = -self.negamax(alpha=-alpha - 1, beta=-alpha, depth=depth - 2, ply_from_root=ply_from_root + 1)
#                     needs_full_search = score > alpha

#                 if (needs_full_search):
#                     # recursive call delegating to the other player
#                     score = -self.negamax(alpha=-beta, beta=-alpha, depth=depth - 1, ply_from_root=ply_from_root + 1)

#                 if (self.debug and ply_from_root == 0):
#                     print(score)
#                     print("====================")

#             self.board.pop()
#             self.bitboard_utils.undo_move(move)
#             self.push_pop_counter -= 1

#             # reset board and piece count
#             if captured_piece is not None:
#                 self.piece_count[util.piece_indices[captured_piece.symbol()]] += 1

#             if (score >= beta):
#                 self.transposition_table.store_evaluation(board=self.board, depth=depth, evaluation=beta, move=best_move, bound=TranspositionBound.LOWER)
#                 return beta

#             if (score > alpha):
#                 evaluation_bound = TranspositionBound.EXACT
#                 alpha = score
#                 best_move = move
#                 if (ply_from_root == 0):
#                     self.search_best_move = best_move
            
#             move_count += 1

#             # pruning techniques that break the move loop
#             if (non_pv and depth <= 4 and captured_piece is None):
#                 quiets_to_check-= 1
#                 if (quiets_to_check == 0):
#                     break

#         self.transposition_table.store_evaluation(board=self.board, depth=depth, evaluation=alpha, move=best_move, bound=evaluation_bound)
#         return alpha

#     def get_move(self):
#         self.max_search_time = self.timer.remaining_time_nanos() // 5
#         self.searching_depth = 1
#         self.push_pop_counter = 0
#         self.color = chess.WHITE if (self.board.ply() % 2 == 0) else chess.BLACK
#         self.search_best_move = None
#         depth_explored = 0
#         while (self.searching_depth <= 10 and self.timer.elapsed_time_nanos() < self.max_search_time // 10):
#             try:
#                 self.negamax(alpha=float('-inf'), beta=float('inf'), depth=self.searching_depth, ply_from_root=0)
                
#                 self.root_best_move = self.search_best_move
#                 depth_explored = self.searching_depth
#             except TimeoutError:
#                 print(f"timeout exception")
#                 break

#             self.searching_depth += 1
        
#         print(f"depth_explored={depth_explored}, move_count={self.num_moves + 1}")
#         self.timer.pretty_print_time_remaining()
        
#         # needed to fix board state after a timeout
#         while(self.push_pop_counter > 0):
#             move = self.board.pop()
#             # only undo non-null moves
#             if (bool(move)):
#                 self.bitboard_utils.undo_move(move)
#             if self.board.is_capture(move):
#                 _, captured_piece = util.get_captured_piece(self.board, move)
#                 self.piece_count[util.piece_indices[captured_piece.symbol()]] += 1
#             self.push_pop_counter -= 1

#         self.num_moves += 1
#         if (self.root_best_move is None):
#             return next(iter(self.board.legal_moves))
#         return self.root_best_move