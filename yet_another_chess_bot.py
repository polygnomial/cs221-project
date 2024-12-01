import chess
import chess.polyglot
from typing import List

from agent import Agent
from piece_count import eval_piece_count
from piece_square_tables import piece_square_table_score
from pawn_shield_storm import eval_pawn_storm
from king_safety import evaluate_king_safety
from mobility import evaluate_mobility
import util


PAWN_VALUE = 100
KNIGHT_VALUE = 200
BISHOP_VALUE = 320
ROOK_VALUE = 500
QUEEN_VALUE = 900

class YetAnotherAgent(Agent):
    def __init__(self, name):
        super().__init__(name)
        self.weights =  {
            "piece_count": 1.0,
            "mobility_assess": 0.0, #added mobility WRT legal moves vs. opponent
            "king_safety": 0.0, #added king safety vs. opponent
            "pawn_storm": 0,
            "piece_square": 0,
        }
        # Configuration
        self.max_search_time = None
        self.searching_depth = 0
        self.last_score = 0
        self.push_pop_counter = 0
        self.num_moves = 0
        self.search_best_move = None

    def featureExtractor(self, piece_count: List[int], board: chess.Board):
        return {
            "piece_count": eval_piece_count(self.piece_count),
            "mobility_assess": evaluate_mobility(board) if self.weights.get("mobility_assess", 0) > 0 else 0, #how many legal moves availabl assuming >0
            "king_safety": evaluate_king_safety(board) if self.weights.get("king_safety", 0) > 0 else 0,
            "pawn_storm": eval_pawn_storm(board) if self.weights["pawn_storm"] > 0.0 else 0,
            "piece_square": piece_square_table_score(board, piece_count) if self.weights["piece_square"] > 0.0 else 0
        }

    # simple evaluation function
    def eval_board(self, board: chess.Board, piece_count: List[int]):
        return util.dotProduct(self.featureExtractor(piece_count, board), self.weights)

    def negamax(self, depth: int, alpha: float, beta: float):
        if self.timer.elapsed_time_nanos() >= self.max_search_time and self.searching_depth > 1:
            raise TimeoutError()
        if (self.board.is_stalemate() or self.board.is_insufficient_material()):
            return (0, None)
        if (self.board.is_checkmate()):
            score = float('inf') if (self.board.outcome().winner == chess.WHITE) else float('-inf')
            return (score, None)
        if (depth == 0):
            return (self.eval_board(self.board, self.piece_count), None)
        
        moves = list(self.board.legal_moves)
        scores = []

        for move in moves:
            # if the move is a capture, decrement the count of the captured piece
            captured_piece = None
            if self.board.is_capture(move):
                _, captured_piece = util.get_captured_piece(self.board, move)
                self.piece_count[util.piece_indices[captured_piece.symbol()]] -= 1

            self.bitboard_utils.make_move(move)
            self.board.push(move)
            self.push_pop_counter += 1

            # recursive call delegating to the other player
            score, _ = self.negamax(depth=depth - 1, alpha=-beta, beta=-alpha)

            self.board.pop()
            self.bitboard_utils.undo_move(move)
            self.push_pop_counter -= 1

            # reset board and piece count
            if captured_piece is not None:
                self.piece_count[util.piece_indices[captured_piece.symbol()]] += 1

            if (self.board.turn == chess.WHITE): # max
                if (score >= beta): #prune
                    return (score, move)
                if (score > alpha):
                    alpha = score
            else: #min
                if (score <= alpha): #prune
                    return (score, move)
                if (score < beta):
                    beta = score
            scores.append(score)

        bestScore = max(scores) if self.board.turn == chess.WHITE else min(scores)
        if (self.board.turn == self.color):
            self.search_best_move = moves[scores.index(bestScore)]
        return (bestScore, moves[scores.index(bestScore)])

    def get_move(self):
        if (self.max_search_time is None):
            self.max_search_time = self.timer.remaining_time_nanos() // 70
        if (self.num_moves > 70):
            self.max_search_time = self.timer.remaining_time_nanos() // 4
        self.searching_depth = 1
        self.push_pop_counter = 0
        self.color = chess.WHITE if (self.board.ply() % 2 == 0) else chess.BLACK
        self.search_best_move = None

        while (self.searching_depth <= 10 and self.timer.elapsed_time_nanos() < self.max_search_time):
            try:
                self.negamax(alpha=float('-inf'), beta=float('inf'), depth=self.searching_depth)
                
                self.root_best_move = self.search_best_move
            except TimeoutError:
                print("timeout exception")
                self.timer.pretty_print_time_remaining()
                break

            self.searching_depth += 1
        
        # needed to fix board state after a timeout
        while(self.push_pop_counter > 0):
            move = self.board.pop()
            # only undo null moves
            if (bool(move)):
                self.bitboard_utils.undo_move(move)
            self.push_pop_counter -= 1

        self.num_moves += 1
        return self.root_best_move