import chess

class RepetitionTable:
    def __init__(self):
        self.count = 0
        self.size = 1024
        self.hashes = [0 for _ in range(self.size -1)]
        self.start_indices = [0 for _ in range(self.size)]
    
    def initialize_table(self, board: chess.Board):
        zobrist_key = chess.polyglot.zobrist_hash(board)
        initial_hashes = [zobrist_key]
        self.count = 1

        for i in range(self.count):
            self.hashes[i] = initial_hashes[i]
            self.start_indices[i] = 0
        self.start_indices[self.count] = 0

    def push(self, hash_value, reset):
        if (self.count < len(self.hashes)):
            self.hashes[self.count] = hash_value
            self.start_indices[self.count + 1] = self.count if(reset) else self.start_indices[self.count]
            self.count += 1

    def try_pop(self):
        self.count = max(0, self.count - 1)

    def contains(self, h):
        for i in range(self.start_indices[self.count], self.count - 1):
            if (self.hashes[i] == h):
                return True
        return False