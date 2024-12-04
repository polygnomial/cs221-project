import chess
from bitboards import BitboardUtils
from repetitions import RepetitionTable
from timer import Timer

import util

class Agent():
    def __init__(self, name: str):
        self.piece_count = None
        self.board = None
        self.bitboard_utils = None
        self.repetition_table = None
        self.timer = None
        self._name = name
    
    def initialize(self, board: chess.Board, bitboard_utils: BitboardUtils, repetition_table: RepetitionTable, timer: Timer):
        self.piece_count = util.initialize_piece_count(board)
        self.board = board
        self.bitboard_utils = bitboard_utils
        self.repetition_table = repetition_table
        self.timer = timer

    def name(self) -> str:
        return self._name

    def get_move(self):
        raise Exception("Not yet implemented")
