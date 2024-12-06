import chess
from copy import deepcopy
from enum import Enum
from typing import Optional
from agent import Agent
from collections import defaultdict
from audio import ChessAudio
from graphics import ChessGraphics
from multiprocessing import Pool, cpu_count
import random
import sys
from typing import List, Tuple
from util import read_positions
from tqdm import tqdm
from bitboards import BitboardUtils
from repetitions import RepetitionTable
from yet_another_chess_bot import YetAnotherAgent
import util
from timer import Timer
from minimax_agent import MiniMaxAgent
from alpha_beta_agent import AlphaBetaAgent

class Variant(Enum):
    Manual = 1
    TestAgents = 2
    Watch = 3
    Fresh = 4
    TestSingleMove = 5

class ChessGame():
    def __init__(self, player1: Optional[Agent] = None, player2: Optional[Agent] = None, useGraphics: bool = True, useAudio: bool = True, startingFen: Optional[str] = None):
        self.player1 = player1
        self.player2 = player2
        self.board = chess.Board()
        if (startingFen):
            self.board.set_fen(startingFen)
        
        self.bitboard_utils = BitboardUtils(board=self.board)
        self.bitboard_utils.initialize_bitboards()

        self.repetition_table = RepetitionTable()
        self.repetition_table.initialize_table(self.board)

        self.audio = ChessAudio() if (useAudio or player1 is None or player2 is None) else None
        # self.player1_timer = Timer(300000000000) # 5 min
        # self.player2_timer = Timer(300000000000) # 5 min
        # self.player1_timer = Timer(600000000000) # 10 min
        # self.player2_timer = Timer(600000000000) # 10 min
        self.player1_timer = Timer(1200000000000) # 20 min
        self.player2_timer = Timer(1200000000000) # 20 min
        self.graphics = ChessGraphics(
            board=self.board,
            bitboard_utils=self.bitboard_utils,
            repetition_table=self.repetition_table,
            audio=self.audio) if (useGraphics or player1 is None or player2 is None) else None
        if (self.player1 is not None):
            self.player1.initialize(
                board=self.board,
                bitboard_utils=self.bitboard_utils,
                repetition_table=self.repetition_table,
                timer=self.player1_timer)
        if (self.player2 is not None):
            self.player2.initialize(
                board=self.board,
                bitboard_utils=self.bitboard_utils,
                repetition_table=self.repetition_table,
                timer=self.player2_timer)

    def play_audio(self, move: chess.Move):
        if (self.audio is None):
            return
        if (self.board.is_capture(move)):
            self.audio.play_capture()
        else:
            self.audio.play_move()

    def add_new_board_hash_to_repetition_table(self, reset: bool):
        zobrist_key = chess.polyglot.zobrist_hash(self.board)
        self.repetition_table.push(zobrist_key, reset)

    def run(self):
        status = True
        winner = None
        while (status):
            if (self.graphics is not None):
                self.graphics.draw_game()
            if self.board.turn == chess.WHITE:
                self.player1_timer.resume()
                if (self.player1 is not None):
                    move = self.player1.get_move()
                    self.play_audio(move)
                    move_piece_type = util.get_piece_type_int(util.get_moved_piece(self.board, move))
                    is_capture_move = self.board.is_capture(move)
                    self.bitboard_utils.make_move(move)
                    self.board.push(move)
                    if (self.graphics is not None):
                        self.graphics.update_last_move(move)
                    self.add_new_board_hash_to_repetition_table(reset=is_capture_move or move_piece_type == 1)
                else:
                    status = self.graphics.capture_human_interaction()
                self.player1_timer.pause()
                if (self.player1_timer.did_buzz()):
                    print("buzz buzz")
                    status = False
                    winner = chess.BLACK
            else:
                self.player2_timer.resume()
                if (self.player2 is not None):
                    move = self.player2.get_move()
                    self.play_audio(move)
                    move_piece_type = util.get_piece_type_int(util.get_moved_piece(self.board, move))
                    is_capture_move = self.board.is_capture(move)
                    self.bitboard_utils.make_move(move)
                    self.board.push(move)
                    if (self.graphics is not None):
                        self.graphics.update_last_move(move)
                    self.add_new_board_hash_to_repetition_table(reset=is_capture_move or move_piece_type == 1)
                else:
                    status = self.graphics.capture_human_interaction()
                self.player2_timer.pause()
                if (self.player2_timer.did_buzz()):
                    print("buzz buzz")
                    status = False
                    winner = chess.WHITE
            
            if (self.board.can_claim_fifty_moves()):
                status = False
                return None
        
            if self.board.outcome() != None:
                status = False
                winner = self.board.outcome().winner
        if (winner == None):
            return None
        if (chess.WHITE == winner):
            print("white won")
            return self.player1.name() if (self.player1 is not None) else 'WHITE'
        else:
            print("black won")
            return self.player2.name() if (self.player2 is not None) else 'BLACK'

# Simulate a single game and return the winner
def simulate_game(data):
    opening,fen,player1,player2 = data
    result = ChessGame(
        player1=player1,
        player2=player2,
        useGraphics=False,
        useAudio=False,
        startingFen=fen).run()
    return (opening, result, player1)

def aggregate(positions: List[Tuple[str, str]]):
    opening_map = defaultdict(lambda: list())
    for opening, fen in positions:
        opening_map[opening].append(fen)
    return opening_map

def testAgents():
    num_games = 4
    num_chunks = 4
    assert num_games % num_chunks == 0
    num_games //= num_chunks
    numWorkers = cpu_count()  # Adjust this to the number of CPU cores you want to use

    print(numWorkers)
    
    # agent1 = RandomAgent("RandAgent1")
    # agent2 = RandomAgent("RandAgent2")
    # agent1 = MinimaxAgentWithPieceSquareTables("psquaretables", depth=2)
    # agent2 = MiniMaxAgent("mma", depth=2)

    # agent1 = MiniMaxAgent("mma", depth=2)
    # agent1 = MinimaxAgentWithPieceSquareTables("psquaretables", depth=2)
    # agent2 = KingSafetyAndMobility("with_King_safety_and_mobility", depth=2)

    agent1 = lambda: MiniMaxAgent(depth=2, name="MinimaxAgent")
    agent2 = lambda: YetAnotherAgent()

    agent1_name = agent1().name()
    agent2_name = agent2().name()
    
    chunks = random.sample(range(1, 21), num_chunks)

    positions_to_play = []
    for chunk in chunks:
        positions = read_positions(f"positions/unprocessed/chunk_{chunk}.txt")
        opening_map = aggregate(positions)
        unique_opening_positions = []
        for opening in opening_map:
            fen = random.choice(opening_map[opening])
            unique_opening_positions.append((opening, fen))

        # changed this so that each opening is played twice, once with each agent as white
        opening_positions = random.sample(unique_opening_positions, num_games)
        for opening, fen in opening_positions:
            positions_to_play.append((opening, fen, agent2(), agent1()))
            positions_to_play.append((opening, fen, agent1(), agent2()))
        
    # Run games in parallel with a progress bar and running tally
    total_games = len(positions_to_play)
    games_played = 0
    winnerMap = defaultdict(lambda : {'WHITE': 0, 'BLACK': 0})

    with Pool(processes=numWorkers) as pool:
        with tqdm(total=total_games, desc=f"Simulating {total_games} games") as pbar:
            for opening, winner, player1 in pool.imap_unordered(simulate_game, positions_to_play):
                # Update running tally
                games_played += 1

                if winner == player1.name():
                    winnerMap[winner]['WHITE'] += 1    
                elif winner is not None:
                    winnerMap[winner]['BLACK'] += 1
                if winner is None:
                    if player1.name() == agent1_name:
                        winnerMap[None]['WHITE'] += 1
                    else:
                        winnerMap[None]['BLACK'] += 1

                # Display running tally in tqdm's description
                a1_wins_w = winnerMap[agent1_name]['WHITE']
                a1_losses_w = winnerMap[agent2_name]['BLACK']
                a1_ties_w = winnerMap[None]['WHITE']
                a1_wins_b = winnerMap[agent1_name]['BLACK']
                a1_losses_b = winnerMap[agent2_name]['WHITE']
                a1_ties_b = winnerMap[None]['BLACK']
                
                pbar.set_postfix_str(f"Agent 1 as white: {a1_wins_w}-{a1_losses_w}-{a1_ties_w}, Agent 1 as black: {a1_wins_b}-{a1_losses_b}-{a1_ties_b}")
                pbar.update(1)

    # Final results
    for winner, count in winnerMap.items():
        if winner is None:
            print(f"{agent1_name} tied {count['WHITE']} as white, {count['BLACK']} as black")
        else:
            print(f"{winner} won {count}/{total_games}")

def runManual():
    ChessGame(player2=YetAnotherAgent(), startingFen="3n2r1/pp3kpp/8/3P4/PPb4b/2P2P1P/5BK1/R6R b - - 0 1").run()

def runSingleGame(startingFen):
    ChessGame(player1=YetAnotherAgent(), player2=MiniMaxAgent(name="minimax_agent", depth=2), startingFen=startingFen).run()

def runVariant(variant: Variant):
    match variant:
        case Variant.Manual:
            return runManual()
        case Variant.TestAgents:
            return testAgents()
        case Variant.Watch:
            chunk = random.randrange(1, 21)
            positions = read_positions(f"positions/unprocessed/chunk_{chunk}.txt")
            _, fen = random.choice(positions)
            return runSingleGame(startingFen=fen)
        case Variant.Fresh:
            return runSingleGame()
        case Variant.TestSingleMove:
            board = chess.Board()
            board.set_fen("3n2r1/pp3kpp/8/3P4/PPb4b/2P1BP1P/6K1/R6R w - - 0 1")
            bitboard_utils = BitboardUtils(board=board)
            bitboard_utils.initialize_bitboards()
            repetition_table = RepetitionTable()
            repetition_table.initialize_table(board)
            agent = YetAnotherAgent(debug=True)
            agent_timer = Timer(600000000000) # 10 min
            agent.initialize(
                board=board,
                bitboard_utils=bitboard_utils,
                repetition_table=repetition_table,
                timer=agent_timer)
            agent_timer.resume()
            move = agent.get_move()
            print("AGENT PICKED")
            print(move)
        case _:
            return f"Unknown variant {variant}"

if __name__ == "__main__":
    numArgs = len(sys.argv) 
    assert numArgs == 1 or numArgs == 3
    variant = Variant.TestAgents
    if (numArgs == 3):
        assert sys.argv[1] == '-v'
        if (sys.argv[2] == "manual"):
            variant = Variant.Manual
        elif (sys.argv[2] == "watch"):
            variant = Variant.Watch
        elif (sys.argv[2] == "fresh"):
            variant = Variant.Fresh
        elif (sys.argv[2] == "test"):
            variant = Variant.TestAgents
        elif (sys.argv[2] == 'testSingleMove'):
            variant = Variant.TestSingleMove
        else:
            raise Exception(f"Unknown argument {sys.argv[2]}")
    print(runVariant(variant))
