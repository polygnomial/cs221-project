import chess

# Piece-square tables for opening and endgame stages
# Opening tables

'''def eval_pawn_shield(piece_count, board):
    
    From https://www.chessprogramming.org/King_Safety
    When the king has castled, it is important to preserve pawns next to it,
    in order to protect it against the assault. Generally speaking, it is best
    to keep the pawns unmoved or possibly moved up one square. The lack of a
    shielding pawn deserves a penalty,
    even more so if there is an open file next to the king.
      
'''
def eval_pawn_storm(board):
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

    white_storm_score = eval_side_storm(
        board,
        chess.WHITE,
        black_king_square)

    black_storm_score = eval_side_storm(
        board,
        chess.BLACK,
        white_king_square)

    pawn_storm_score = white_storm_score - black_storm_score

    return pawn_storm_score

def eval_side_storm(board, storm_king_colour, enemy_king_square):
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

    for file in files_to_check: #Looping through columns
        for rank in range(8): #Looping through rows 0-7
            
            square = chess.square(file, rank) #Checking each square for each Row X Column            
            piece = board.piece_at(square)

            #For the piece on the board having same color and is also a pawn :) 
            if piece and piece.piece_type == chess.PAWN and piece.color == storm_king_colour:
                square_score = 0  # Initializing square_score for each pawn

                # More pawns of the color on the board means more score :)
                if storm_king_colour == chess.WHITE:
                    square_score += rank + 1  # White pawns higher rank = higher score
                else:
                    square_score += 8 - rank  # Black pawns lower rank = higher score
                
                # Score higher when pawn is closer to the enemy king
                distance_to_enemy_king = chess.square_distance(square, enemy_king_square)
                '''
                Bonus allocation : 0.2 multiplier is chosen because
                Pawn = 1 point , Knight/Bishop = 3 points , Rook = 5 points, Queen = 9 points
                So with 0.2:
                    A pawn 1 square from king gets 1.2 bonus points i.e.similar to pawn value
                    A pawn 4 squares away gets 0.6 bonus points i.e. half pawn value
                '''
                if distance_to_enemy_king == 0: 
                    square_score += 1.4 
                elif distance_to_enemy_king == 1:
                    square_score += 1.2 
                elif distance_to_enemy_king == 2:
                    square_score += 1.0
                elif distance_to_enemy_king == 3:
                    square_score += 0.8
                elif distance_to_enemy_king == 4:
                    square_score += 0.6 
                elif distance_to_enemy_king == 5:
                    square_score += 0.4
                elif distance_to_enemy_king == 6:
                    square_score += 0.2
                else: # distance is 7
                    square_score += 0.0
                
                storm_score += square_score # Adding this pawns score to total storm score
    return storm_score