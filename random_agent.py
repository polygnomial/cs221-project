import random

from agent import Agent

class RandomAgent(Agent):

    def get_move(self):
        return random.choice(list(self.board.legal_moves))