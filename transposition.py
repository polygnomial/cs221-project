import chess
import chess.polyglot
from dataclasses import dataclass

@dataclass
class TranspositionEntry:
    key: int = 0
    evaluation: int = 0
    move: chess.Move = None
    depth: int = 0

class TranspositionTable:
    def __init__(self):
        self.lookup_failed = -1
        self.exact = 0
        self.lower_bound = 1
        self.upper_bound = 2
        self.num_entries = 100000
        self.count = self.num_entries
        self.entries = [TranspositionEntry() for _ in range(self.count)]

    def clear(self):
        for i in range(len(self.num_entries)):
            self.entries[i] = TranspositionEntry()
    
    def index(self, board: chess.Board):
        zobrist_key = chess.polyglot.zobrist_hash(board)
        return zobrist_key % self.count

    def try_get_stored_move(self, board: chess.Board):
        return self.entries[self.index(board)].move

    def store_evaluation(self, board: chess.Board, depth: int, evaluation: int, move: chess.Move):
        index = self.index(board)
        self.entries[index] = TranspositionEntry(key=chess.polyglot.zobrist_hash(board), evaluation=evaluation, move=move, depth=depth)

    def get_entry(self, board: chess.Board):
        return self.entries[self.index(board)]