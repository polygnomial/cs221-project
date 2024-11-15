# importing required librarys
import pygame
import chess
import math
import time
from agent import min_max_agent, random_agent
from collections import defaultdict


#initialise display
X = 800
Y = 800
scrn = pygame.display.set_mode((X, Y))
pygame.init()

#basic colours
WHITE = (255, 255, 255)
GREY = (128, 128, 128)
YELLOW = (204, 204, 0)
BLUE = (50, 255, 255)
BLACK = (0, 255, 0)
TAN = (236, 218, 185)
BROWN = (174, 138, 104)

#load piece images
pieces = {
    'p': pygame.image.load('images/black_pawn.png').convert_alpha(),
    'n': pygame.image.load('images/black_knight.png').convert_alpha(),
    'b': pygame.image.load('images/black_bishop.png').convert_alpha(),
    'r': pygame.image.load('images/black_rook.png').convert_alpha(),
    'q': pygame.image.load('images/black_queen.png').convert_alpha(),
    'k': pygame.image.load('images/black_king.png').convert_alpha(),
    'P': pygame.image.load('images/white_pawn.png').convert_alpha(),
    'N': pygame.image.load('images/white_knight.png').convert_alpha(),
    'B': pygame.image.load('images/white_bishop.png').convert_alpha(),
    'R': pygame.image.load('images/white_rook.png').convert_alpha(),
    'Q': pygame.image.load('images/white_queen.png').convert_alpha(),
    'K': pygame.image.load('images/white_king.png').convert_alpha(),
}

for key in pieces:
    pieces[key] = pygame.transform.scale(pieces[key], (100, 100))

def promote_pawn(board, start_square, end_square):
    """Handles promotion of a pawn with user choice."""
    promotion_choice = input("Promote to (q)ueen, (r)ook, (b)ishop, or (n)ight? ")
    promotion_piece = {
        'q': chess.QUEEN,
        'r': chess.ROOK,
        'b': chess.BISHOP,
        'n': chess.KNIGHT
    }.get(promotion_choice.lower(), chess.QUEEN)  # Default to queen

    # Create the promotion move
    move = chess.Move(start_square, end_square, promotion=promotion_piece)
    if move in board.legal_moves:
        board.push(move)
        print("Pawn promoted!")
    else:
        print("Illegal move.")

# Example move with promotion from e7 to e8
# promote_pawn(board, chess.E7, chess.E8)
# print(board)

def update(scrn,board):
    '''
    updates the screen basis the board class
    '''
    
    for i in range(64):
        piece = board.piece_at(i)
        if piece == None:
            pass
        else:
            scrn.blit(pieces[str(piece)],((i%8)*100,700-(i//8)*100))

    pygame.display.flip()

def drawBoard():
    for i in range(8):
        for j in range(8):
            color = TAN if (i%2 == j%2) else BROWN
            scrn.fill(color, pygame.Rect(j*100, i*100, 100, 100))

def main(BOARD):

    '''
    for human vs human game
    '''
    #name window
    pygame.display.set_caption('Chess')

    drawBoard()

    #variable to be used later
    index_moves = []

    status = True
    while (status):
        #update screen
        update(scrn,BOARD)

        for event in pygame.event.get():
     
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                status = False

            # if mouse clicked
            if event.type == pygame.MOUSEBUTTONDOWN:
                #remove previous highlights
                # scrn.fill(BROWN)
                #get position of mouse
                pos = pygame.mouse.get_pos()

                #find which square was clicked and index of it
                square = (math.floor(pos[0]/100),math.floor(pos[1]/100))
                index = (7-square[1])*8+(square[0])
                
                # if we are moving a piece
                if index in index_moves: 
                    
                    move = moves[index_moves.index(index)]
                    
                    BOARD.push(move)

                    drawBoard()

                    #reset index and moves
                    index=None
                    index_moves = []
                    
                    
                # show possible moves
                else:
                    #check the square that is clicked
                    piece = BOARD.piece_at(index)
                    #if empty pass
                    if piece == None:
                        
                        pass
                    else:
                        
                        #figure out what moves this piece can make
                        all_moves = list(BOARD.legal_moves)
                        moves = []
                        for m in all_moves:
                            if m.from_square == index:
                                
                                moves.append(m)

                                t = m.to_square

                                TX1 = 100*(t%8)
                                TY1 = 100*(7-t//8)

                                
                                #highlight squares it can move to
                                pygame.draw.circle(scrn,BLUE,(TX1+50,TY1+50),10)
                        
                        index_moves = [a.to_square for a in moves]
     
    # deactivates the pygame library
        if BOARD.outcome() != None:
            print(BOARD.outcome())
            status = False
            print(BOARD)
    pygame.quit()

def main_one_agent(agent,agent_color):
    #initialise chess board
    board = chess.Board()
    
    '''
    for agent vs human game
    color is True = White agent
    color is False = Black agent
    '''
    
    #make background black
    scrn.fill(BLACK)
    #name window
    pygame.display.set_caption('Chess')
    
    #variable to be used later
    index_moves = []

    status = True
    while (status):
        drawBoard()
        #update screen
        update(scrn,board)
        
     
        if board.turn==agent_color:
            board.push(agent(board))
            scrn.fill(BLACK)

        else:

            for event in pygame.event.get():
         
                # if event object type is QUIT
                # then quitting the pygame
                # and program both.
                if event.type == pygame.QUIT:
                    status = False

                # if mouse clicked
                if event.type == pygame.MOUSEBUTTONDOWN:
                    #reset previous screen from clicks
                    scrn.fill(BLACK)
                    #get position of mouse
                    pos = pygame.mouse.get_pos()

                    #find which square was clicked and index of it
                    square = (math.floor(pos[0]/100),math.floor(pos[1]/100))
                    index = (7-square[1])*8+(square[0])
                    
                    # if we have already highlighted moves and are making a move
                    if index in index_moves: 
                        
                        move = moves[index_moves.index(index)]
                        #print(BOARD)
                        #print(move)
                        board.push(move)
                        index=None
                        index_moves = []
                        
                    # show possible moves
                    else:
                        
                        piece = board.piece_at(index)
                        
                        if piece == None:
                            
                            pass
                        else:

                            all_moves = list(board.legal_moves)
                            moves = []
                            for m in all_moves:
                                if m.from_square == index:
                                    
                                    moves.append(m)

                                    t = m.to_square

                                    TX1 = 100*(t%8)
                                    TY1 = 100*(7-t//8)

                                    
                                    #highlight squares it can move to
                                    pygame.draw.circle(scrn,BLUE,(TX1+50,TY1+50),10)
                            #print(moves)
                            index_moves = [a.to_square for a in moves]
     
    # deactivates the pygame library
        if board.outcome() != None:
            print(board.outcome())
            status = False
            print(board)
    pygame.quit()

def main_two_agent(agent1,agent2):
    '''
    for agent vs agent game
    
    '''
    #initialise chess board
    board = chess.Board()
    agent_color1 = board.turn
  
    #make background black
    scrn.fill(BLACK)
    #name window
    pygame.display.set_caption('Chess')
    
    #variable to be used later

    status = True
    while (status):
        drawBoard()
        #update screen
        update(scrn,board)
        
        if board.turn==agent_color1:
            board.push(agent1(board))

        else:
            board.push(agent2(board))

        scrn.fill(BLACK)
            
        for event in pygame.event.get():
     
            # if event object type is QUIT
            # then quitting the pygame
            # and program both.
            if event.type == pygame.QUIT:
                status = False
     
    # deactivates the pygame library
        if board.outcome() != None:
            print(board.outcome())
            status = False
            print(board)
            drawBoard()
            #update screen
            update(scrn,board)
            time.sleep(5)
    pygame.quit()

def main_headless_two_agent(agent1,agent2):
    #initialise chess board
    board = chess.Board()
    agent_color1 = board.turn
    '''
    for agent vs agent game
    
    '''

    status = True
    winner = None
    while (status):
        if board.turn==agent_color1:
            board.push(agent1(board))
        else:
            board.push(agent2(board))
     
        if board.outcome() != None:
            print(board.outcome())
            status = False
            print(board)
            winner = board.outcome().winner
    return winner

numGames = 50
for i in range(numGames):
    winnerMap = defaultdict(lambda: 0)
    winner = main_headless_two_agent(min_max_agent(2), random_agent)
    winnerMap[winner] += 1

for winner in winnerMap:
    print(f"{winner} won {winnerMap[winner]}/{numGames}")

# main_one_agent(b,play_min_maxN,b.turn)