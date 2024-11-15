import chess
import math
import pygame

class ChessGraphics():
    def __init__(self, board: chess.Board, dimension: int = 800):
        #basic colors
        self.WHITE = (255, 255, 255)
        self.GREY = (128, 128, 128)
        self.YELLOW = (204, 204, 0)
        self.BLUE = (50, 255, 255)
        self.BLACK = (0, 255, 0)
        self.TAN = (236, 218, 185)
        self.BROWN = (174, 138, 104)

        self.board = board
        
        self.screen = self.initialize_screen(dimension)
        #make background black
        self.screen.fill(self.BLACK)
        #name window
        pygame.display.set_caption('Chess')
        self.pieces = self.initialize_pieces()
    
    def initialize_screen(self, dimension: int):
        screen = pygame.display.set_mode((dimension, dimension))
        pygame.init()
        return screen
    
    def initialize_pieces(self):
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
        return pieces
    
    def draw_game(self):
        self.draw_board()
        self.draw_pieces()

    def draw_board(self):
        for i in range(8):
            for j in range(8):
                color = self.TAN if (i%2 == j%2) else self.BROWN
                self.screen.fill(color, pygame.Rect(j*100, i*100, 100, 100))
    
    def draw_pieces(self):
        '''
        updates the screen basis the board class
        '''
        
        for i in range(64):
            piece = self.board.piece_at(i)
            if piece == None:
                pass
            else:
                self.screen.blit(self.pieces[str(piece)],((i%8)*100,700-(i//8)*100))

        pygame.display.flip()
    
    def highlight_square(self, destination: int):
        destination_x = 100*(destination%8)
        destination_y = 100*(7-destination//8)
        pygame.draw.circle(self.screen,self.BLUE,(destination_x + 50, destination_y + 50),10)

    def capture_human_interaction(self):
        moves = dict()
        while(True):
            for event in pygame.event.get():
            
                # if event object type is QUIT
                # then quitting the pygame
                # and program both.
                if event.type == pygame.QUIT:
                    return False

                # if mouse clicked
                if event.type == pygame.MOUSEBUTTONDOWN:
                    #reset previous screen from clicks
                    self.screen.fill(self.BLACK)
                    #get position of mouse
                    pos = pygame.mouse.get_pos()

                    #find which square was clicked and index of it
                    square = (math.floor(pos[0]/100),math.floor(pos[1]/100))
                    index = (7-square[1])*8+(square[0])

                    if index in moves: # make the move
                        self.board.push(moves[index])
                        return True
                    else: # show possible moves
                        piece = self.board.piece_at(index)
                        if piece is not None:
                            for move in list(self.board.legal_moves):
                                if move.from_square == index:
                                    moves[move.to_square] = move
                                    self.highlight_square(move.to_square)