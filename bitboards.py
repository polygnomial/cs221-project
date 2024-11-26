import chess
from gmpy2 import bit_scan1

import util

class BitboardUtils:
    def __init__(self, board: chess.Board):
        self.board = board

        self.piece_bitboards = [0 for _ in range(15)]
        self.color_bitboards = [0, 0]
        
        self.white_king_square = 0
        self.black_king_square = 0

        self.all_pieces_bitboard = 0

        self.rook_directions = [ (-1, 0), (1, 0), (0, 1), (0, -1) ]
        self.bishop_directions = [ (-1, 1), (1, 1), (1, -1), (-1, -1) ]

        self.rook_shifts = [
            52, 52, 52, 52, 52, 52, 52, 52,
            53, 53, 53, 54, 53, 53, 54, 53,
            53, 54, 54, 54, 53, 53, 54, 53,
            53, 54, 53, 53, 54, 54, 54, 53,
            52, 54, 53, 53, 53, 53, 54, 53,
            52, 53, 54, 54, 53, 53, 54, 53,
            53, 54, 54, 54, 53, 53, 54, 53,
            52, 53, 53, 53, 53, 53, 53, 52
        ]
        self.bishop_shifts = [
            58, 60, 59, 59, 59, 59, 60, 58,
            60, 59, 59, 59, 59, 59, 59, 60,
            59, 59, 57, 57, 57, 57, 59, 59,
            59, 59, 57, 55, 55, 57, 59, 59,
            59, 59, 57, 55, 55, 57, 59, 59,
            59, 59, 57, 57, 57, 57, 59, 59,
            60, 60, 59, 59, 59, 59, 60, 60,
            58, 60, 59, 59, 59, 59, 59, 58
        ]

        self.rook_magics = [
            468374916371625120,     18428729537625841661,   2531023729696186408,    6093370314119450896,
            13830552789156493815,   16134110446239088507,   12677615322350354425,   5404321144167858432,
            2111097758984580,       18428720740584907710,   17293734603602787839,   4938760079889530922,
            7699325603589095390,    9078693890218258431,    578149610753690728,     9496543503900033792,
            1155209038552629657,    9224076274589515780,    1835781998207181184,    509120063316431138,
            16634043024132535807,   18446673631917146111,   9623686630121410312,    4648737361302392899,
            738591182849868645,     1732936432546219272,    2400543327507449856,    5188164365601475096,
            10414575345181196316,   1162492212166789136,    9396848738060210946,    622413200109881612,
            7998357718131801918,    7719627227008073923,    16181433497662382080,   18441958655457754079,
            1267153596645440,       18446726464209379263,   1214021438038606600,    4650128814733526084,
            9656144899867951104,    18444421868610287615,   3695311799139303489,    10597006226145476632,
            18436046904206950398,   18446726472933277663,   3458977943764860944,    39125045590687766,
            9227453435446560384,    6476955465732358656,    1270314852531077632,    2882448553461416064,
            11547238928203796481,   1856618300822323264,    2573991788166144,       4936544992551831040,
            13690941749405253631,   15852669863439351807,   18302628748190527413,   12682135449552027479,
            13830554446930287982,   18302628782487371519,   7924083509981736956,    4734295326018586370
        ]
        self.bishop_magics = [
            16509839532542417919,   14391803910955204223,   1848771770702627364,    347925068195328958,
            5189277761285652493,    3750937732777063343,    18429848470517967340,   17870072066711748607,
            16715520087474960373,   2459353627279607168,    7061705824611107232,    8089129053103260512,
            7414579821471224013,    9520647030890121554,    17142940634164625405,   9187037984654475102,
            4933695867036173873,    3035992416931960321,    15052160563071165696,   5876081268917084809,
            1153484746652717320,    6365855841584713735,    2463646859659644933,    1453259901463176960,
            9808859429721908488,    2829141021535244552,    576619101540319252,     5804014844877275314,
            4774660099383771136,    328785038479458864,     2360590652863023124,    569550314443282,
            17563974527758635567,   11698101887533589556,   5764964460729992192,    6953579832080335136,
            1318441160687747328,    8090717009753444376,    16751172641200572929,   5558033503209157252,
            17100156536247493656,   7899286223048400564,    4845135427956654145,    2368485888099072,
            2399033289953272320,    6976678428284034058,    3134241565013966284,    8661609558376259840,
            17275805361393991679,   15391050065516657151,   11529206229534274423,   9876416274250600448,
            16432792402597134585,   11975705497012863580,   11457135419348969979,   9763749252098620046,
            16960553411078512574,   15563877356819111679,   14994736884583272463,   9441297368950544394,
            14537646123432199168,   9888547162215157388,    18140215579194907366,   18374682062228545019
        ]

        self.rook_mask = [self.create_movement_mask(i, True) for i in range(64)]
        self.rook_attacks = [self.create_table(i, True, self.rook_magics[i], self.rook_shifts[i]) for i in range(64)]

        self.bishop_mask = [self.create_movement_mask(i, False) for i in range(64)]
        self.bishop_attacks = [self.create_table(i, False, self.bishop_magics[i], self.bishop_shifts[i]) for i in range(64)]
    
    def initialize_bitboards(self):
        for square in self.board.piece_map():
            piece = self.board.piece_map()[square]
            color_index = 0 if (piece.color == chess.WHITE) else 1
            piece_type = util.piece_type_map[piece.symbol()]
            piece_index = piece_type | (color_index << 3)
            if (piece_type == 6): # king
                if (color_index == 0):
                    self.white_king_square = square
                else:
                    self.black_king_square = square
            elif (piece_type < 0 or piece_type > 6):
                raise NotImplemented(f"{piece.symbol()} not supported yet by bitboard")
            self.piece_bitboards[piece_index] = self.set_square(self.piece_bitboards[piece_index], square)
            self.color_bitboards[color_index] = self.set_square(self.color_bitboards[color_index], square)

        self.all_pieces_bitboard = self.color_bitboards[0] | self.color_bitboards[1]

    def undo_move(self, move: chess.Move):
        self.make_move(move = move, undo_move = True)

    def make_move(self, move: chess.Move, undo_move: bool = False):
        piece = self.board.piece_at(move.from_square)
        color_index = 0 if (piece.color == chess.WHITE) else 1
        piece_type = util.piece_type_map[piece.symbol()]
        piece_index = piece_type | (color_index << 3)

        self.piece_bitboards[piece_index] = self.toggle_square(self.piece_bitboards[piece_index], move.from_square)
        self.piece_bitboards[piece_index] = self.toggle_square(self.piece_bitboards[piece_index], move.to_square)
        self.color_bitboards[color_index] = self.toggle_square(self.color_bitboards[color_index], move.from_square)
        self.color_bitboards[color_index] = self.toggle_square(self.color_bitboards[color_index], move.to_square)

        if (self.board.is_capture(move)):
            captured_piece_square, captured_piece = util.get_captured_piece(self.board, move)
            captured_piece_color_index = 0 if (captured_piece.color == chess.WHITE) else 1
            captured_piece_type = util.piece_type_map[captured_piece.symbol()]
            captured_piece_index = captured_piece_type | (captured_piece_color_index << 3)
            self.piece_bitboards[captured_piece_index] = self.toggle_square(self.piece_bitboards[captured_piece_index], captured_piece_square)
            self.color_bitboards[captured_piece_color_index] = self.toggle_square(self.color_bitboards[captured_piece_color_index], captured_piece_square)

        move_type = util.get_move_type(self.board, move)
        if (move_type == util.MoveType.CASTLING):
            is_kingside = move.to_square == chess.G1 or move.to_square == chess.G8
            if (piece.color == chess.WHITE):
                self.white_king_square = move.from_square if (undo_move) else move.to_square
            else:
                self.black_king_square = move.from_square if (undo_move) else move.to_square
            castling_rook_from_index = move.to_square + 1  if (is_kingside) else move.to_square - 2
            castling_rook_to_index = move.to_square - 1 if (is_kingside) else move.to_square + 1
            rook_piece_index = 4 | (color_index << 3)

            self.piece_bitboards[rook_piece_index] = self.toggle_square(self.piece_bitboards[rook_piece_index], castling_rook_from_index)
            self.piece_bitboards[rook_piece_index] = self.toggle_square(self.piece_bitboards[rook_piece_index], castling_rook_to_index)
            self.color_bitboards[color_index] = self.toggle_square(self.color_bitboards[color_index] , castling_rook_from_index)
            self.color_bitboards[color_index] = self.toggle_square(self.color_bitboards[color_index] , castling_rook_to_index)

        if (move_type == util.MoveType.PAWN_PROMOTE_TO_BISHOP
            or move_type == util.MoveType.PAWN_PROMOTE_TO_KNIGHT 
            or move_type == util.MoveType.PAWN_PROMOTE_TO_ROOK
            or move_type == util.MoveType.PAWN_PROMOTE_TO_QUEEN):
            pawn_piece_index = 1 | (color_index << 3)
            promotion_piece_index = color_index << 3
            if (move_type == util.MoveType.PAWN_PROMOTE_TO_KNIGHT):
                promotion_piece_index |= 2
            elif (move_type == util.MoveType.PAWN_PROMOTE_TO_BISHOP):
                promotion_piece_index |= 3
            elif (move_type == util.MoveType.PAWN_PROMOTE_TO_ROOK):
                promotion_piece_index |= 4
            elif (move_type == util.MoveType.PAWN_PROMOTE_TO_QUEEN):
                promotion_piece_index |= 5

            self.piece_bitboards[pawn_piece_index] = self.toggle_square(self.piece_bitboards[pawn_piece_index], move.to_square)
            self.piece_bitboards[promotion_piece_index] = self.toggle_square(self.piece_bitboards[promotion_piece_index], move.to_square)

        self.all_pieces_bitboard = self.color_bitboards[0] | self.color_bitboards[1]

    def undo_move(self, move: chess.Move):
        self.make_move(move) # toggling the same bits should yield in undoing the bit boards

    def toggle_square(self, bitboard, square):
        return bitboard ^ (1 << square)
    
    def set_square(self, bitboard, square):
        return bitboard | (1 << square)

    def create_table(self, square, is_rook, magic, left_shift):
        num_bits = 64 - left_shift
        lookup_size = 1 << num_bits
        table = [0 for _ in range(lookup_size)]

        movementMask = self.create_movement_mask(square, is_rook)
        blockerPatterns = self.create_all_blocker_bitboards(movementMask)

        for pattern in blockerPatterns:
            index = (pattern * magic) >> left_shift
            moves = self.legal_move_bitboard_from_blockers(square, pattern, is_rook)
            table[index] = moves

        return table

    def create_all_blocker_bitboards(self, movementMask):
        move_square_indices = []
        for i in range(64):
            if (((movementMask >> i) & 1) == 1):
                move_square_indices.append(i)

        num_patterns = 1 << len(move_square_indices)
        blocker_bitboards = [0 for _ in range(num_patterns)]

        for pattern_index in range(num_patterns):
            for bit_index in range(len(move_square_indices)):
                bit = (pattern_index >> bit_index) & 1
                blocker_bitboards[pattern_index] = blocker_bitboards[pattern_index] | (bit << move_square_indices[bit_index])

        return blocker_bitboards

    def legal_move_bitboard_from_blockers(self, square_index, blocker_bitboard, is_rook):
        bitboard = 0

        directions = self.rook_directions if(is_rook) else self.bishop_directions
        startCoord = (chess.square_file(square_index), chess.square_rank(square_index))

        for direction in directions:
            for dist in range(1, 8):
                coord_file = startCoord[0] + direction[0] * dist
                coord_rank = (startCoord[1] + direction[1] * dist)
                coord_square = coord_rank * 8 + coord_file

                if (self.is_valid_square(coord_file, coord_rank)):
                    bitboard = bitboard | (1 << coord_square)
                    if (self.contains_square(blocker_bitboard, coord_square)):
                        break
                else:
                    break

        return bitboard

    def create_movement_mask(self, square_index, is_rook):
        mask = 0
        directions = self.rook_directions if(is_rook) else self.bishop_directions
        startCoord = (chess.square_file(square_index), chess.square_rank(square_index))

        for direction in directions:
            for dist in range(1, 8):
                coord_file = startCoord[0] + direction[0] * dist
                coord_rank = (startCoord[1] + direction[1] * dist)
                coord_square = coord_rank * 8 + coord_file
                
                next_coord_file = startCoord[0] + direction[0] * (dist + 1)
                next_coord_rank = startCoord[1] + direction[1] * (dist + 1)

                if (self.is_valid_square(next_coord_file, next_coord_rank)):
                    mask = mask | (1 << coord_square)
                else:
                    break
        return mask

    def contains_square(self, bitboard, square):
        return ((bitboard >> square) & 1) != 0

    def is_valid_square(self, file, rank):
        file >= 0 and file < 8 and rank >= 0 and rank < 8

    def clear_lsb(self, bitboard):
        return bitboard & (bitboard - 1)

    def get_lsb_index(self, bitboard):
        return bit_scan1(bitboard)

    def get_bishop_attacks(self, square, blockers):
        mask = self.bishop_mask[square]
        magic = self.bishop_magics[square]
        shift = self.bishop_shifts[square]
        key = ((blockers & mask) * magic) >> shift
        return self.bishop_attacks[square][key]

    def get_rook_attacks(self, square, blockers):
        mask = self.rook_mask[square]
        magic = self.rook_magics[square]
        shift = self.rook_shifts[square]
        key = ((blockers & mask) * magic) >> shift
        return self.rook_attacks[square][key]

    def get_queen_attacks(self, square, blockers):
        return self.get_rook_attacks(square, blockers) | self.get_rook_attacks(square, blockers)

    def get_slider_attacks(self, piece_type, square):
        if (piece_type == 3): # bishop
            return self.get_bishop_attacks(square, self.all_pieces_bitboard)
        elif (piece_type == 4): # rook
            return self.get_rook_attacks(square, self.all_pieces_bitboard)
        elif (piece_type == 5): # queen
            return self.get_queen_attacks(square, self.all_pieces_bitboard)
        else:
            return 0
    
    def get_piece_bitboard(self, piece_type: int, piece_is_white: bool):
        piece_index = piece_type if (piece_is_white) else 8 | piece_type
        return self.piece_bitboards[piece_index]