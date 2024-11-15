import random
import chess
import pdb
from piece_square_tables import piece_square_table_score

#an agent that moves randommly
def random_agent(BOARD):
    return random.choice(list(BOARD.legal_moves))

scoring= {
    'p': -1, # Black pawn
    'n': -3, # Black knight
    'b': -3, # Black bishop (should be slightly more valuable than knight ideally for better evaluation)
    'r': -5, # Black rook
    'q': -9, # Black queen
    'k': 0,  # Black king
    'P': 1,  # White pawn
    'N': 3,  # White knight
    'B': 3,  # White bishop (should be slightly more valuable than knight ideally for better evaluation)  
    'R': 5,  # White rook
    'Q': 9,  # White queen
    'K': 0,  # White king
}


# initialize and return a piece count dictionary
def initialize_piece_count(board):
    
    piece_count = {'P': 0, 'N': 0, 'B': 0, 'R': 0, 'Q': 0, 'K': 0,
               'p': 0, 'n': 0, 'b': 0, 'r': 0, 'q': 0, 'k': 0}

    pieces = board.piece_map()
    for piece in pieces.values():
        piece_count[str(piece)] += 1

    return piece_count

# simple evaluation function
def eval_board(piece_count, board):
    score = 0
    for piece, count in piece_count.items():
        score += count * scoring[piece]

    storm_score = eval_pawn_storm(piece_count, board)
    score += storm_score

    score += piece_square_table_score(board, piece_count)

    return score


def min_maxN(board, depth, alpha, beta, piece_count):
    if (board.is_stalemate() or board.is_insufficient_material()):
        return (0, None)
    if (board.is_checkmate()):
        score = float('inf') if (board.outcome().winner == chess.WHITE) else float('-inf')
        return (score, None)
    if (depth == 0):
        return (eval_board(piece_count, board), None)
    
    moves = list(board.legal_moves)
    scores = []

    for move in moves:
        # if the move is a capture, decrement the count of the captured piece
        was_capture = False
        if board.is_capture(move):
            captured_piece = str(board.piece_at(move.to_square))
            piece_count[captured_piece] -= 1
            was_capture = True

        board.push(move)

        # recursive call delegating to the other player
        score, _ = min_maxN(board,depth-1, alpha, beta, piece_count)

        # reset board and piece count
        board.pop()
        if was_capture:
            piece_count[captured_piece] += 1

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
    piece_count = initialize_piece_count(board)
    return min_maxN(board,N, float('-inf'), float('inf'), piece_count)[1]

'''def eval_pawn_shield(piece_count, board):
    
    From https://www.chessprogramming.org/King_Safety
    When the king has castled, it is important to preserve pawns next to it,
    in order to protect it against the assault. Generally speaking, it is best
    to keep the pawns unmoved or possibly moved up one square. The lack of a
    shielding pawn deserves a penalty,
    even more so if there is an open file next to the king.
      
'''
def eval_pawn_storm(piece_count, board):
    '''
    From https://www.chessprogramming.org/King_Safety
    If the enemy pawns are near to the king, there might be a threat of
    opening a file, even if the pawn shield is intact. Penalties for storming
    enemy pawns must be lower than penalties for (semi)open files,
    otherwise the pawn storm might backfire, resulting in a blockage.
    '''
    white_king_square = None
    black_king_square = None
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.symbol() == 'K':
                white_king_square = square #Getting white kind position
            elif piece.symbol() == 'k':
                black_king_square = square #Getting black kind position

    if white_king_square is None or black_king_square is None:
        return 0 #If either king is missing dont proceed


    white_storm_score = eval_side_storm(board, chess.WHITE, piece_count, white_king_square,
                                        black_king_square)

    black_storm_score = eval_side_storm(board, chess.BLACK, piece_count, black_king_square,
                                        white_king_square)

    pawn_storm_score = white_storm_score - black_storm_score
    #pdb.set_trace()  

    return pawn_storm_score

def eval_side_storm(board, storm_king_colour, piece_count, own_king_square, enemy_king_square):
    '''
    '''

    storm_score = 0

    '''
    Sides where queenside castling occurs and where a queenside pawn storm would happen
    queen side files = [0, 1, 2] (a, b, c files)

    Sides from where kingside castling occurs and where a kingside pawn storm would typically happen
    king side files = [5, 6, 7] (f, g, h files) 
    '''
    files_to_check = [] #Use for determining files to check based on the location of the enemy king
    enemy_king_file = chess.square_file(enemy_king_square)    #Location of enemy kind

    if enemy_king_file <= 3: # If enemy king is on a-d files (queen side_
        files_to_check = [0, 1, 2]  # a, b, c files
    else:          # If enemy king is on e-h files (kingside)
        files_to_check = [5, 6, 7]  # f, g, h files
        
    if storm_king_colour == chess.WHITE:
        pawn_count = piece_count['P']  #White 
    else:
        pawn_count = piece_count['p']  #Black 

    for file in files_to_check: #Looping through columns
        for rank in range(8): #Looping through rows 0-7
            
            square = chess.square(file, rank) #Checking each square for each Row X Column            
            piece = board.piece_at(square)

            #For the piece on the board having same color and is also a pawn :) 
            if piece and piece.piece_type == chess.PAWN and piece.color == storm_king_colour:
                square_score = 0  # Initializing square_score for each pawn

                # More pawns of the color on the board means more score :)
                if storm_king_colour == chess.WHITE:
                    square_score = square_score + rank + 1  # White pawns higher rank = higher score
                else:
                    square_score = square_score + 8 - rank  # Black pawns lower rank = higher score
                
                # Score higher when pawn is closer to the enemy king
                distance_to_enemy_king_bonus = 0
                distance_to_enemy_king = chess.square_distance(square, enemy_king_square)
                '''
                Bonus allocation : 0.2 multiplier is chosen because
                Pawn = 1 point , Knight/Bishop = 3 points , Rook = 5 points, Queen = 9 points
                So with 0.2:
                    A pawn 1 square from king gets 1.2 bonus points i.e.similar to pawn value
                    A pawn 4 squares away gets 0.6 bonus points i.e. half pawn value
                '''
                if distance_to_enemy_king == 0: 
                    square_score = square_score + 1.4 
                elif distance_to_enemy_king == 1:
                    square_score = square_score + 1.2 
                elif distance_to_enemy_king == 2:
                    square_score = square_score + 1.0
                elif distance_to_enemy_king == 3:
                    square_score = square_score + 0.8
                elif distance_to_enemy_king == 4:
                    square_score = square_score + 0.6 
                elif distance_to_enemy_king == 5:
                    square_score = square_score + 0.4
                elif distance_to_enemy_king == 6:
                    square_score = square_score + 0.2
                elif distance_to_enemy_king == 7:
                    square_score = square_score + 0.0
                else:
                    square_score = square_score + 0.0
                
                storm_score += square_score # Adding this pawns score to total storm score

    #pdb.set_trace()        
    return storm_score


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
