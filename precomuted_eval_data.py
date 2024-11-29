import chess
from typing import List

import util

class PrecomputedEvaluationData:
    def __init__(self):
        self.file_offsets = [-1, 0, 1]
        self.pawn_shield_squares_white: List[List[int]] = [[] for _ in range(64)]
        self.pawn_shield_squares_black: List[List[int]] = [[] for _ in range(64)]
        for square in range(64):
            self.create_pawn_shield_square(square)

    def create_pawn_shield_square(self, square: int):
        shield_indices_white: List[int] = []
        shield_indices_black: List[int] = []

        rank = chess.square_rank(square)
        file = util.clamp(chess.square_file(square), 1, 6)

        for file_offset in self.file_offsets:
            self.add_if_valid(util.rank_and_file_to_square(file + file_offset, rank + 1), shield_indices_white)
            self.add_if_valid(util.rank_and_file_to_square(file + file_offset, rank - 1), shield_indices_black)

        for file_offset in self.file_offsets:
            self.add_if_valid(util.rank_and_file_to_square(file + file_offset, rank + 2), shield_indices_white)
            self.add_if_valid(util.rank_and_file_to_square(file + file_offset, rank - 2), shield_indices_black)

        self.pawn_shield_squares_white[square] = shield_indices_white
        self.pawn_shield_squares_black[square] = shield_indices_black

    def add_if_valid(self, square: int, shield_index_list: List[int]):
        if (square >= 0 and square < 64):
            shield_index_list.append(square)