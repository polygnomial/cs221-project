import chess
from typing import Optional
from agent import Agent, MiniMaxAgent, RandomAgent, MinimaxAgentWithPieceSquareTables
from collections import defaultdict
from graphics import ChessGraphics
from multiprocessing import Pool, cpu_count
import random
from typing import List, Tuple
from util import read_positions
from tqdm import tqdm
import os, sys

class ChessGame():
    def __init__(self, player1: Optional[Agent] = None, player2: Optional[Agent] = None, useGraphics: bool = True, startingFen: Optional[str] = None):
        self.player1 = player1
        self.player2 = player2
        self.board = chess.Board()
        if (startingFen):
            self.board.set_fen(startingFen)
        self.graphics = ChessGraphics(board=self.board) if (useGraphics or player1 is None or player2 is None) else None
        if (self.player1 is not None):
            self.player1.initialize(board=self.board)
        if (self.player2 is not None):
            self.player2.initialize(board=self.board)

    def run(self):
        status = True
        winner = None
        while (status):
            if (self.graphics is not None):
                self.graphics.draw_game()
            if self.board.turn == chess.WHITE:
                if (self.player1 is not None):
                    self.board.push(self.player1.get_move())
                else:
                    status = self.graphics.capture_human_interaction()
            else:
                if (self.player2 is not None):
                    self.board.push(self.player2.get_move())
                else:
                    status = self.graphics.capture_human_interaction()
        
            if self.board.outcome() != None:
                # print(self.board.outcome())
                status = False
                # print(self.board)
                winner = self.board.outcome().winner
        if (winner == None):
            return None
        if (chess.WHITE == winner):
            return self.player1.name() if (self.player1 is not None) else "white"
        else:
            return self.player2.name() if (self.player2 is not None) else "black"

# Simulate a single game and return the winner
def simulate_game(data):
    opening,fen,player1,player2 = data
    result = ChessGame(
        player1=player1,
        player2=player2,
        useGraphics=False,
        startingFen=fen).run()
    return (opening, result)

def aggregate(positions: List[Tuple[str, str]]):
    opening_map = defaultdict(lambda: list())
    for opening, fen in positions:
        opening_map[opening].append(fen)
    return opening_map

if __name__ == "__main__":
    num_games = 2
    numWorkers = cpu_count()  # Adjust this to the number of CPU cores you want to use

    chunks = random.sample(range(1, 21), 2)

    agent1 = MiniMaxAgent(depth=2)
    agent2 = MinimaxAgentWithPieceSquareTables(depth=2)

    winnerMap = defaultdict(int)
    unique_opening_positions = []
    positions_to_play = []
    for chunk in chunks:
        positions = read_positions(f"positions/unprocessed/chunk_{chunk}.txt")
        opening_map = aggregate(positions)
        for opening in opening_map:
            fen = random.choice(opening_map[opening])
            unique_opening_positions.append((opening, fen, agent1, agent2))
        positions_to_play += random.sample(unique_opening_positions, num_games)

        # swap agents so they take turns playing white and black per chunk
        agent1, agent2 = agent2, agent1

    # Run games in parallel with a progress bar and running tally
    total_games = len(positions_to_play)
    games_played = 0

    with Pool(processes=numWorkers) as pool:
        with tqdm(total=total_games, desc=f"Simulating {total_games} games") as pbar:
            for opening, winner in pool.imap_unordered(simulate_game, positions_to_play):
                # Update running tally
                games_played += 1
                winnerMap[winner] += 1

                # Display running tally in tqdm's description
                tie_count = winnerMap[None]
                wins = defaultdict(lambda : 0)
                for k, v in winnerMap.items():
                    if k is not None:
                        wins[k] = v
                
                win_summary = ", ".join(f"{winner}: {count}" for winner, count in wins.items())
                pbar.set_postfix_str(f"Agent 1 won {wins[agent1.name()]}/{games_played}, Agent 2 won {wins[agent2.name()]}/{games_played}, Ties: {tie_count}/{games_played}")
                pbar.update(1)

    # Final results
    for winner, count in winnerMap.items():
        if winner is None:
            print(f"Agents tied {count}/{total_games}")
        else:
            print(f"{winner} won {count}/{total_games}")
