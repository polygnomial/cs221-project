import tkinter as tk
import chess
import chess.svg
from PIL import Image, ImageTk
import io
import cairosvg

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PyChess GUI")

        self.board = chess.Board()
        
        # Create a label to display the chess board
        self.board_canvas = tk.Label(self.root)
        self.board_canvas.pack()

        # Text entry for moves
        self.move_entry = tk.Entry(self.root)
        self.move_entry.pack()
        self.move_entry.bind("<Return>", self.make_move)

        # Label to show game status
        self.status_label = tk.Label(self.root, text="Your move!")
        self.status_label.pack()

        self.update_board()

    def make_move(self, event=None):
        move = self.move_entry.get().strip()
        self.move_entry.delete(0, tk.END)
        try:
            chess_move = self.board.parse_san(move)
            self.board.push(chess_move)
            self.update_board()
            
            if self.board.is_checkmate():
                self.status_label.config(text="Checkmate!")
            elif self.board.is_stalemate():
                self.status_label.config(text="Stalemate!")
            elif self.board.is_insufficient_material():
                self.status_label.config(text="Draw by insufficient material!")
            elif self.board.is_check():
                self.status_label.config(text="Check!")
            else:
                self.status_label.config(text="Your move!")

        except ValueError:
            self.status_label.config(text="Invalid move, try again.")

    def update_board(self):
        svg_data = chess.svg.board(self.board).encode("utf-8")
        png_data = cairosvg.svg2png(bytestring=svg_data)
        image = Image.open(io.BytesIO(png_data))
        photo = ImageTk.PhotoImage(image)

        self.board_canvas.config(image=photo)
        self.board_canvas.image = photo

# Initialize the GUI application
root = tk.Tk()
chess_gui = ChessGUI(root)
root.mainloop()
