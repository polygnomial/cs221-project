import chess

from agent import Agent
from piece_count import eval_piece_count
import util

class MiniMaxAgent(Agent):
    def __init__(self, name, depth: int):
        super().__init__(name)
        self.depth = depth
        self.weights =  {
            "piece_count": 1.0,
        }

    def featureExtractor(self):
        return {
            "piece_count": eval_piece_count(self.piece_count),
        }

    # simple evaluation function
    def eval_board(self):
        return util.dotProduct(self.featureExtractor(), self.weights)

    def min_maxN(self, depth: int):
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
            score, _ = self.min_maxN(depth=depth - 1)

            self.board.pop()
            self.bitboard_utils.undo_move(move)

            # reset self.board and piece count
            if captured_piece is not None:
                self.piece_count[util.piece_indices[captured_piece.symbol()]] += 1

            scores.append(score)

        bestScore = max(scores) if self.board.turn == chess.WHITE else min(scores)
        return (bestScore, moves[scores.index(bestScore)])

    def get_move(self):
        _, move = self.min_maxN(depth=self.depth*2)
        return move