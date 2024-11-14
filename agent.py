import random
import chess

#an agent that moves randommly
def random_agent(BOARD):
    return random.choice(list(BOARD.legal_moves))

scoring=  {
    'p': -10,    # Black pawn
    'n': -33,    # Black knight
    'b': -38,    # Black bishop (slightly more valuable than knight)
    'r': -50,    # Black rook
    'q': -90,    # Black queen
    'k': 0,      # Black king
    'P': 10,     # White pawn
    'N': 33,     # White knight
    'B': 38,     # White bishop (slightly more valuable than knight)
    'R': 50,     # White rook
    'Q': 90,     # White queen
    'K': 0       # White king
    }

#simple evaluation function
'''
def eval_board(board):
    score = 0
    pieces = board.piece_map()
    for key in pieces:
        score += scoring[str(pieces[key])]

    return score
'''
#simple evaluation function + King Pawn shield
def eval_board(BOARD):
    score = 0
    WHITE = True
    BLACK = False
    pieces = BOARD.piece_map()
    # pdb.set_trace()  # Set a breakpoint here
    for key in pieces:
        score += scoring[str(pieces[key])]
    
    # Adding Pawn shield evaluation to the simple evaluation function
    white_king_eval = BOARD.king(WHITE)
    black_king_eval = BOARD.king(BLACK)
    
    if white_king_eval is not None:
        score += king_pawn_shield(BOARD, white_king_eval, True)
    
    if black_king_eval is not None:
        score -= king_pawn_shield(BOARD, black_king_eval, False)
    
    return score


def min_maxN(board, depth, alpha, beta):
    if (board.is_stalemate() or board.is_insufficient_material()):
        return (0, None)
    if (board.is_checkmate()):
        score = float('inf') if (board.outcome().winner == chess.WHITE) else float('-inf')
        return (score, None)
    if (depth == 0):
        return (eval_board(board), None)
    
    moves = list(board.legal_moves)
    scores = []

    for move in moves:
        # push our move and then delegate to the adversary
        board.push(move)
        score, _ = min_maxN(board,depth-1, alpha, beta)
        board.pop()
        if (board.turn == chess.WHITE): # max
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

    bestScore = max(scores) if board.turn == chess.WHITE else min(scores)
    return (bestScore, moves[scores.index(bestScore)])
        
# a simple wrapper function as the display only gives one imput , BOARD
def play_min_maxN(board):
    N=2*2 # depth 2 but multiply by 2 to ensure both players play per depth
    return min_maxN(board,N, float('-inf'), float('inf'))[1]


# import chess.polyglot
# import chess

# BOARD = chess.Board()
# #opening book
# reader = chess.polyglot.open_reader('baron30.bin')

# #search the opening book for this game state
# opening_move = reader.get(BOARD)

# #if no move is found
# if opening_move == None:
#     move = None
# #if move is found
# else:
#     move = opening_move.move

# print(move)

#Return higher positive score for shield in front of the king
def king_pawn_shield(board, king_square, is_white):
    shield_score = 0
    king_column = king_square % 8  #King Column
    king_row = king_square // 8 #Kind Row
    
    # Check left, center, right in front of the king
    for file in range(max(0, king_column - 1), min(8, king_column + 2)): #Confirming we are not going off the board while checking        
        if is_white: # For white king : check one and two ranks above as White pawns can only move upward and hence white pawns protect their king from ranks above them
            check_rows = [king_row + 1, king_row + 2] if king_row < 6 else [king_row + 1]
        else: # For black king : check one and two ranks below as Black pawns can only move downward and hence black pawns protect their king from ranks below them
            check_rows = [king_row - 1, king_row - 2] if king_row > 1 else [king_row - 1]
        
        for row in check_rows:
            if 0 <= row < 8:  # Ensuring row is on the board
                square = row * 8 + file 
                piece = board.piece_at(square)
                if piece and piece.symbol() == ('P' if is_white else 'p'):
                    shield_score += 10  # Adding points for each protecting pawn
    
    return shield_score



