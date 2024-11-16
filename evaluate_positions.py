import chess.engine
from multiprocessing import Pool
from typing import List, Tuple
from util import read_positions

def write_positions(file_path: str, positions: List[Tuple[str, str]]):
    with open(file_path, 'w') as file:
        for opening, fen in positions:
            file.write(opening + '\n')
            file.write(fen + '\n')

def process_positions(chunk):
    print(f"Chunk {chunk}: Reading positions...")
    
    positions = read_positions(f"positions/unprocessed/chunk_{chunk}.txt")
    
    print(f"Chunk {chunk}: Finished reading positions...beginning evaluation...")
    
    engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
    board = chess.Board()

    equivalent_positions = []
    count = 0
    for opening, fen in positions:
        board.set_fen(fen)
        score = engine.analyse(board, chess.engine.Limit(time=1))["score"].relative
        if(not score.is_mate() and abs(score.score()) <= 20):
            equivalent_positions.append((opening, fen))
        count += 1
        if (count % 100 == 0):
            print(f"Chunk {chunk}: Finished evaluating {count} positions...{len(positions)-count} more to go...")
    
    print(f"Chunk {chunk}: Finished evaluating positions...outputting to file...")
    
    engine.quit()

    write_positions(f"positions/processed/chunk_{chunk}.txt", equivalent_positions)


    print(f"Chunk {chunk}: Finished writing equivalent positions to file...")

if __name__ == "__main__":
    with Pool(processes=10) as pool:
        pool.map(process_positions, range(1, 21))

