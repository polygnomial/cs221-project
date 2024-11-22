import chess

def evaluate_king_safety(board: chess.Board):
    """
    float positive for safer white king, negative for safe black king
    """
    white_king_safety = _king_safety(board, chess.WHITE)
    black_king_safety = _king_safety(board, chess.BLACK)
    return white_king_safety - black_king_safety #positive = white king safer

def _king_safety(board: chess.Board, color: chess.Color):
    """
            Evaluate the safety of the king of the specified color.

    Args:
        board (chess.Board): The current board state.
        color (chess.Color): The color of the king to evaluate.

    Returns:
        float: The safety score for the king.
    """
    king_square = board.king(color) #where is king

    if king_square is None: #shouldn't happen, just to be safe
        return float('-inf') if color == chess.WHITE else float('inf')
    safety_score = 0
    safety_score += _pawn_shield(board, king_square, color) #protect the king
    safety_score += _open_lines_to_king(board, king_square, color) #dangerous paths of access to king
    safety_score += _enemy_pieces_near_king(board, king_square, color) #dangerous nearby pieces
    #safety_score += self._castling_status(board, color) #castling option if needed adds safety
    safety_score += _king_exposure(board, king_square, color) #overall exposure risk for king
    return safety_score

def _pawn_shield(board: chess.Board, king_square: int, color: chess.Color):
    """
    Positive if pawns are protecting the king, otherwise negative.
    """
    shield_squares = _get_pawn_shield_squares(king_square, color)
    pawn_shield_score = 0
    for square in shield_squares:
        if not (0 <= square <= 63):  # Check if square index is valid
            continue
        piece = board.piece_at(square)
        if piece is not None and piece.piece_type == chess.PAWN and piece.color == color:
            pawn_shield_score += 0.5
        else:
            pawn_shield_score -= 0.5  # Penalize no shield
    return pawn_shield_score

def _get_pawn_shield_squares(king_square: int, color: chess.Color):
    """
    get cells comprising potential pawn shield
    """
    rank = chess.square_rank(king_square)
    file = chess.square_file(king_square)

    shield_squares = []
#add rank if self color is in front of king; o/w penalize
    if color == chess.WHITE:
        rank += 1
    else:
        rank -= 1
    for df in [-1, 0, 1]:
        f = file + df
        if 0 <= f <= 7 and 0 <= rank <= 7: #try to bound file and rank; had trouble here
            shield_squares.append(chess.square(f, rank))
    return shield_squares

def _open_lines_to_king(board: chess.Board, king_square: int, color: chess.Color):
    #open files and diags to king. Might not be realistic?

    open_files_penalty = 0
    file = chess.square_file(king_square)
    friendly_pawns_in_file = any(
        board.piece_at(chess.square(file, r)) == chess.Piece(chess.PAWN, color)
        for r in range(8)
    )
    if not friendly_pawns_in_file: #penalize no friendly pawns in file
        #should add additional rules and biases for non-pawns in file.
        open_files_penalty -= 0.5
    return open_files_penalty

def _enemy_pieces_near_king(board: chess.Board, king_square: int, color: chess.Color):
    enemy_color = not color
    threat_penalty = 0
    # Get the squares adjacent to the king
    king_area = chess.SquareSet(chess.BB_KING_ATTACKS[king_square]) #from 64 bit integer BB all squares to which king can move from current and convert to chess.SquareSet for iteration over squares...only squares to which king can move
    for square in king_area:
        piece = board.piece_at(square)
        if piece is not None and piece.color == enemy_color:
            threat_penalty -= 0.3
    return threat_penalty

#def _castling_status(self, board: chess.Board, color: chess.Color):
#    if board.is_castled(color):
#        return 0.5  # Reward for castling
#    else:
#        return -0.5  # Penalty for not castling

def _king_exposure(board: chess.Board, king_square: int, color: chess.Color):
    """
    penalize getting too deep into the board; probably should vary depending upon game state
    """
    exposure_penalty = 0
    rank = chess.square_rank(king_square)
    if color == chess.WHITE and rank >= 5:
        exposure_penalty -= 0.5  # Penalty for white king being too advanced
    elif color == chess.BLACK and rank <= 2:
        exposure_penalty -= 0.5  # Penalty for black king being too advanced
    return exposure_penalty