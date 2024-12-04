import chess
import math
import os, sys
from audio import ChessAudio
from bitboards import BitboardUtils
import util
from repetitions import RepetitionTable

# disable printing of "hello from the pygame community" message
sys.stdout = open(os.devnull, 'w')
import pygame
sys.stdout = sys.__stdout__

class ChessGraphics():
    def __init__(self, board: chess.Board, bitboard_utils: BitboardUtils, repetition_table: RepetitionTable, audio: ChessAudio, dimension: int = 800):
        #basic colors
        self.WHITE = (255, 255, 255)
        self.GREY = (128, 128, 128)
        self.TAN_YELLOW = (207, 209, 134)
        self.BROWN_YELLOW = (170, 162, 86)
        self.TAN_GREEN = (138, 151, 111)
        self.BROWN_GREEN = (107, 111, 70)
        self.BLACK = (0, 0, 0)
        self.TAN = (236, 218, 185)
        self.BROWN = (174, 138, 104)

        self.board = board
        self.bitboard_utils = bitboard_utils
        self.repetition_table = repetition_table
        self.audio = audio
        
        self.screen = self.initialize_screen(dimension*1.2)
        #make background black
        self.screen.fill(self.BLACK)
        #name window
        pygame.display.set_caption('Chess')
        self.pieces = self.initialize_pieces()
        self.moves = dict()
        self.last_move = None
    
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
        self.highlight_squares()
        self.draw_last_move()
        self.draw_pieces()
    
    def draw_rect(self, color: chess.Color, row: int, col: int):
        pygame.draw.rect(self.screen, color,(row*100, col*100, 100, 100))

    def get_position(self, square: int):
        row = square%8
        col = 7-square//8
        return (row, col)
    
    def update_last_move(self, move):
        self.last_move = move
    
    def draw_last_move_square(self, square):
        if (square not in self.moves):
            row, col = self.get_position(square)
            color = self.TAN_YELLOW if (row%2 == col%2) else self.BROWN_YELLOW
            self.draw_rect(color, row, col)

    def draw_last_move(self):
        if (self.last_move):
            self.draw_last_move_square(self.last_move.from_square)
            self.draw_last_move_square(self.last_move.to_square)
    
    def highlight_squares(self):
        for square in self.moves:
            row, col = self.get_position(square)
            if (self.board.piece_at(square) is None):
                color = self.TAN_GREEN if (row%2 == col%2) else self.BROWN_GREEN
                pygame.draw.circle(self.screen, color, (100*row + 50, 100*col + 50), 10)
            else:
                circ_color = self.TAN if (row%2 == col%2) else self.BROWN
                rect_color = self.TAN_GREEN if (row%2 == col%2) else self.BROWN_GREEN
                self.draw_rect(rect_color, row, col)
                pygame.draw.circle(self.screen, circ_color, (100*row + 50, 100*col + 50), 50, width=0)

    def draw_board(self):
        for i in range(8):
            for j in range(8):
                color = self.TAN if ((i+j) % 2 == 0) else self.BROWN
                pygame.draw.rect(self.screen, color,(j*100, i*100, 100, 100))
    
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

    def play_audio(self, move: chess.Move):
        if (self.audio is None):
            return
        if (self.board.is_capture(move)):
            self.audio.play_capture()
        else:
            self.audio.play_move()

    def capture_human_interaction(self):
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

                if index in self.moves: # make the move
                    move = self.moves[index]
                    self.play_audio(move)
                    move_piece_type = util.get_piece_type_int(util.get_moved_piece(self.board, move))
                    is_capture_move = self.board.is_capture(move)
                    self.bitboard_utils.make_move(move)
                    self.board.push(move)
                    # print("human turn complete")
                    # print(self.board)
                    zobrist_key = chess.polyglot.zobrist_hash(self.board)
                    self.repetition_table.push(zobrist_key, is_capture_move or move_piece_type == 1)
                    self.last_move = move
                    self.moves = dict()
                else: # show possible moves
                    self.moves = dict()
                    piece = self.board.piece_at(index)
                    if piece is not None:
                        for move in list(self.board.legal_moves):
                            if move.from_square == index:
                                self.moves[move.to_square] = move
        return True