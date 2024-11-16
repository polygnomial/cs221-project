import chess.pgn
import random
import math

pgn = open("chess_games.pgn")

positions = []

end_of_file = False
count = 1
while (count <= 20):
    game = chess.pgn.read_game(pgn)
    if (game is None):
        end_of_file = True
    else:
        moves = game.mainline_moves()
        board = game.board()

        plyToPlay = math.floor(16 + 20 * random.random()) & ~1
        numPlyPlayed = 0

        for move in moves:
            board.push(move)
            numPlyPlayed += 1
            if (numPlyPlayed == plyToPlay):
                fen = board.fen()

        numPiecesInPos = sum(fen.lower().count(char) for char in 'rnbq')
        if numPlyPlayed > plyToPlay + 20 * 2 and numPiecesInPos >= 10:
            positions.append(game.headers['Opening'])
            positions.append(fen)
    if (len(positions) % 2000 == 0):
        print(f"Found {len(positions)/2} games")
    if (len(positions) == 20000):
        with open(f"./positions/unprocessed/chunk_{count}.txt", 'w') as file:
            for string in positions:
                file.write(string + '\n')
        count += 1
        positions = []

