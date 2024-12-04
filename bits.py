import chess

import util

class Bits:

    def __init__(self):
        self.all_pieces = 0xFFFFFFFFFFFFFFFF
        self.file_mask = [0 for _ in range(8)]
        self.adjacent_file_masks = [0 for _ in range(8)]
        self.white_passed_pawn_mask = [0 for _ in range(64)]
        self.black_passed_pawn_mask = [0 for _ in range(64)]
        
        for i in range(8):
            self.file_mask[i] = util.a_file << i
            
            left = util.a_file << (i - 1) if (i > 0) else 0
            right = util.a_file << (i + 1) if (i < 7) else 0
            self.adjacent_file_masks[i] = left | right

        for square in range(64):
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            
            adjacent_files = util.a_file << max(0, file - 1) | util.a_file << min(7, file + 1)
            #Passed pawn mask
            white_forward_mask = ~(self.all_pieces >> (64 - 8 * (rank + 1)))
            black_forward_mask = ((1 << 8 * rank) - 1)

            self.white_passed_pawn_mask[square] = (util.a_file << file | adjacent_files) & white_forward_mask
            self.black_passed_pawn_mask[square] = (util.a_file << file | adjacent_files) & black_forward_mask

BITS = Bits()