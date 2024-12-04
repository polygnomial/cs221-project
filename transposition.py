import chess
import chess.polyglot
from dataclasses import dataclass
from enum import Enum

class TranspositionBound(Enum):
    LOWER = 1
    EXACT = 2
    UPPER = 3

@dataclass
class TranspositionEntry:
    key: int = 0
    evaluation: int = 0
    move: chess.Move = None
    depth: int = 0
    bound: TranspositionBound = TranspositionBound.LOWER

class TranspositionTable:
    def __init__(self):
        self.lookup_failed = -1
        self.exact = 0
        self.lower_bound = 1
        self.upper_bound = 2
        self.count = 100000
        self.entries = [None for _ in range(self.count)]

    def clear(self):
        for i in range(len(self.count)):
            self.entries[i] = TranspositionEntry()

    def hash(self, board: chess.Board):
        return chess.polyglot.zobrist_hash(board)
    
    def index(self, hash):
        return hash % self.count

    def store_evaluation(self, board: chess.Board, depth: int, evaluation: int, move: chess.Move, bound: TranspositionBound):
        index = self.index(self.hash(board))
        self.entries[index] = TranspositionEntry(
            key=chess.polyglot.zobrist_hash(board),
            evaluation=evaluation,
            move=move,
            depth=depth,
            bound=bound)

    def get_entry(self, board: chess.Board):
        zobrist_hash = self.hash(board)
        entry = self.entries[self.index(zobrist_hash)]
        return entry if (entry is not None and entry.key == zobrist_hash) else None