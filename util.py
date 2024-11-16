import itertools

def read_positions(file_path: str):
    positions = []
    with open(file_path) as file:
        for opening, fen in itertools.zip_longest(*[file]*2):
            positions.append((opening.strip(), fen.strip()))
    return positions