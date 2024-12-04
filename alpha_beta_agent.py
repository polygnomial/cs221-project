import chess

from agent import Agent
from piece_count import eval_piece_count
from piece_square_tables import piece_square_table_score
from pawn_shield_storm import eval_pawn_storm
from king_safety import evaluate_king_safety
from mobility import evaluate_mobility
import util

class AlphaBetaAgent(Agent):
    def __init__(self, name, depth: int):
        super().__init__(name)
        self.depth = depth
        self.weights =  {
            "piece_count": 1.0,
            "mobility_assess": 0.0, #added mobility WRT legal moves vs. opponent
            "king_safety": 0.0, #added king safety vs. opponent
            "pawn_storm": 0,
            "piece_square": 0,
        }

    def featureExtractor(self):
        return {
            "piece_count": eval_piece_count(self.piece_count),
            "mobility_assess": evaluate_mobility(self.board) if self.weights.get("mobility_assess", 0) > 0 else 0, #how many legal moves availabl assuming >0
            "king_safety": evaluate_king_safety(self.board) if self.weights.get("king_safety", 0) > 0 else 0,
            "pawn_storm": eval_pawn_storm(self.board) if self.weights["pawn_storm"] > 0.0 else 0,
            "piece_square": piece_square_table_score(self.board, self.piece_count) if self.weights["piece_square"] > 0.0 else 0
        }

    # simple evaluation function
    def eval_board(self):
        return util.dotProduct(self.featureExtractor(), self.weights)

    def min_maxN(self, depth: int, alpha: float, beta: float):
        if (self.board.is_stalemate() or self.board.is_insufficient_material()):
            return (0, None)
        if (self.board.is_checkmate()):
            score = float('inf') if (self.board.outcome().winner == chess.WHITE) else float('-inf')
            return (score, None)
        if (depth == 0):
            return (self.eval_board(), None)
        
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

            # recursive call delegating to the other player
            score, _ = self.min_maxN(depth=depth - 1, alpha=alpha, beta=beta)

            self.board.pop()
            self.bitboard_utils.make_move(move)

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
        return (bestScore, moves[scores.index(bestScore)])

    def get_move(self):
        _, move = self.min_maxN(
            depth=self.depth*2,
            alpha=float('-inf'),
            beta=float('inf'))
        return move