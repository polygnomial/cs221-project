
## Double Deep Q-learning Learning Chess Engine Educational Implementation
# Complete (not yet ideal) implementation of a deep reinforcement learning chess engine.
# learns to play  through self-play + competition against increasingly
# difficult opponents, using modern deep learning techniques including # emulation supervised
#learning from grandmaster play (this take a really long time--each episode can be thousands of games--currently set to
#2 games of grandmaster play watching per episode--at full games per episode, 10 episodes of emulation were nearly a million moves and took overnight processing on M3
# - Deep Q-Learning with Experience Replay
# - Residual Neural Networks (ResNets)
# - Long Short-Term Memory (LSTM) for position history plus skips and gradient clips
# - Curriculum Learning with progressive difficulty vs. stockfish--episode 9000 is a checkpoint from
#an earlier non-double Deep Q i ran up to stockfish depth 40. checkpoint_ep9000.pth is a MASSIVE file
#i can create an artifact with more in depth commentary and teaching


import numpy as np  # numerical computations and arrays
import chess  # python-chess library for chess game representation; manipulation/representation of chess boards and moves.
import chess.engine  # Import engine module from python-chess for interacting with chess engines; tools to interact with stockfish etc.
import torch  #  PyTorch library for deep learning
import torch.nn as nn  # Import neural network module from PyTorch; simplifies build/training ANN
import torch.nn.functional as F  # Import functional API from PyTorch for activation functions, etc.
import random  # Import random module for generating random numbers, random sampling
from collections import deque  # double-ended queue simplify FIFO queueing
import torch.optim as optim  # Import optimization algorithms from PyTorch
import math  # Import math module for mathematical functions
from torch.utils.tensorboard import SummaryWriter  # logging training metrics and viz
from torch.utils.data import Dataset, DataLoader, TensorDataset
import os  #  interacting with the operating system
from datetime import datetime  # tracking/logging
import time  # """" """"
# Import time module for time-related functions
import chess.polyglot
import chess.pgn
import threading
import multiprocessing
STOCKFISH_PATH = '/opt/homebrew/bin/stockfish'  # path to the Stockfish chess engine
OPENING_BOOK_PATH = 'data/book.bin'
MASTER_GAMES_PATH = 'data/twic1516.pgn'
# Device configuration
# Check if the Metal Performance Shaders (MPS) backend is available for Apple Silicon devices

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") #PyTorch's device configuration to allow hardware accel (e.g., GPU, Metal on macOS for me on M3).
print(f"Using device: {device}")  # Output the device being used (e.g., 'cpu' or 'mps')

# Create logging directory
# Create a directory for logging training metrics using TensorBoard
log_dir = os.path.join('runs', datetime.now().strftime('%Y%m%d-%H%M%S'))  # Directory named with current date and time
writer = SummaryWriter(log_dir)  # Initialize the TensorBoard writer with the logging directory

# these fxns need to go before residual blocks

def encode_board(board):
    #  converts  current chess board state into a numerical representation to send to neural net.
    # Will creates a 3D NumPy array of shape (8, 8, 12), representing the board's 8x8 grid and 12 possible piece types in tensor.
    board_state = np.zeros((8, 8, 12), dtype=int)  # Initialize the board state array with zeros

    # Mapping of piece symbols to indices in the 12-channel representation
    piece_to_index = {
        'P': 0,  # White Pawn
        'N': 1,  # White Knight
        'B': 2,  # White Bishop
        'R': 3,  # White Rook
        'Q': 4,  # White Queen
        'K': 5,  # White King
        'p': 6,  # Black Pawn
        'n': 7,  # Black Knight
        'b': 8,  # Black Bishop
        'r': 9,  # Black Rook
        'q': 10, # Black Queen
        'k': 11  # Black King
    }

    # Iterate over all squares on the chess board
    for square in chess.SQUARES:
        piece = board.piece_at(square)  # Get the piece at the current square
        if piece:
            x = chess.square_file(square)  # Get the file (column) index of the square (0-7)
            y = chess.square_rank(square)  # Get the rank (row) index of the square (0-7)
            idx = piece_to_index[piece.symbol()]  # Get the corresponding index for the piece type
            board_state[y, x, idx] = 1  # Set the position in the array to 1 where the piece is located
    return board_state  # Return the encoded board state as a NumPy array

def mask_legal_moves(board, q_values):
    #  function masks  illegal moves by setting  Qval neg inf
    #  ensures  agent only considers legal moves during action selection.
    #may not be most efficient because doesn't really reduce the search space (more like a filter post hoc)


    legal_moves = list(board.legal_moves)  # Get a list of all legal moves in the current board state
    legal_indices = [encode_move(m) for m in legal_moves]  # Convert legal moves to their corresponding indices

    # CHANGED: Set illegal moves to -infinity
    masked_q_values = torch.full_like(q_values, -float('inf'))  # Initialize a tensor with -infinity values using "like" to match q val
    masked_q_values[legal_indices] = q_values[legal_indices]  # Replace Q-values of legal moves with  actual values
    return masked_q_values  # Return the masked Q-values tensor now legal only; would be nice to reduce the actual search space but couldn't figure out a means of doing so

def encode_move(move):
    # encodes a chess move into unique integer index. Works really nicely for the operations and the numerical libraries we have
    promotion_offset = 0  # Initialize promotion offset

    if move.promotion:
        # If the move is a promotion, calculate the promotion offset. EVERY PROMOTION IS A UNIQUE OFFSET!
        #dictionary seems to be most efficient here
        promotion_dict = {
            chess.QUEEN: 0,   # Promotion to Queen
            chess.ROOK: 1,    # Promotion to Rook
            chess.BISHOP: 2,  # Promotion to Bishop
            chess.KNIGHT: 3   # Promotion to Knight
        }
        promotion_offset = promotion_dict[move.promotion] * 4096  # Each promotion type gets unique offset
        promotion_offset += 64 * 64  # Offset to account for non-promotion moves
    return move.from_square * 64 + move.to_square + promotion_offset  # Return the unique index for the move

def decode_move(index):
    # decodes a unique integer index back into a chess.Move object; basically just a reversal
    promotion_type = None  # Initialize promotion to None

    if index >= 64 * 64:
        # If the index corresponds to a promotion move, adjust index and determine promotion type
        index -= 64 * 64  # Adjust index for promotion offset
        if index >= 4096 * 3:
            promotion_type = chess.KNIGHT  # Promotion to Knight
            index -= 4096 * 3
        elif index >= 4096 * 2:
            promotion_type = chess.BISHOP  # Promotion to Bishop
            index -= 4096 * 2
        elif index >= 4096:
            promotion_type = chess.ROOK  # Promotion to Rook
            index -= 4096
        else:
            promotion_type = chess.QUEEN  # Promotion to Queen

    from_square = index // 64  # Calculate the from_square index
    to_square = index % 64     # Calculate the to_square index
    return chess.Move(from_square, to_square, promotion=promotion_type)  # Return the decoded move

# Corrected reward function:

def get_reward(board, move, done, is_agent_turn):
    # This function calculates the reward for the agent based on the current board state.
    base_reward = 0.0  # Initialize base reward (not used directly in this implementation)

    if done:
        # If the game is over, determine the reward based on the outcome
        if board.is_checkmate():
            return 4.0 if is_agent_turn else -3.0  # Reward for winning, penalty for losing
        if board.is_stalemate() or board.is_insufficient_material():
            return -1.0  # Slight penalty for draws
        return -0.2  # Small penalty for other game endings

    # Material reward
    piece_values = {
        chess.PAWN: 1,     # Value of a pawn
        chess.KNIGHT: 3,   # Value of a knight
        chess.BISHOP: 3,   # Value of a bishop
        chess.ROOK: 5,     # Value of a rook
        chess.QUEEN: 9,    # Value of a queen
        # N.B. King is not included since the game ends if the king is captured
    }

    # Evaluate material balance
    material_balance = 0  # Initialize material balance
    for square in chess.SQUARES:
        piece = board.piece_at(square)  # Get the piece at the square
        if piece:
            value = piece_values.get(piece.piece_type, 0)  # Get the value of the piece
            if piece.color == chess.WHITE:
                material_balance += value  # Add value for white pieces
            else:
                material_balance -= value  # Subtract value for black pieces

    # Normalize material balance
    normalized_material_balance = 0.01 * material_balance  # Scale down to keep rewards small

    # Center control bonus
    center_squares = {chess.E4, chess.E5, chess.D4, chess.D5}  # Define central squares
    center_control = sum(1 for sq in center_squares if board.piece_at(sq) and board.piece_at(sq).color == chess.WHITE)
    # Count how many central squares are controlled by white pieces

    # Mobility reward (number of legal moves)
    mobility = len(list(board.legal_moves)) if board.turn == chess.WHITE else -len(list(board.legal_moves))
    # Positive mobility for white's turn, negative for black's turn

    # Move number penalty to encourage faster games
    move_penalty = -0.001 * board.fullmove_number  # Small penalty for each move to encourage shorter games

    total_reward = (
        0.1 * material_balance +  # Reward for material advantage
        0.05 * center_control +   # Reward for controlling the center
        0.01 * mobility +         # Reward for mobility
        move_penalty              # Penalty for longer games
    )

    return total_reward if is_agent_turn else -total_reward  # Return the reward based on whose turn it is

class ReplayBuffer:
    def __init__(self, capacity):
        # Initialize the replay buffer with a given capacity
        self.buffer = deque(maxlen=capacity)  # Use deque for efficient appends and pops from both ends

    def push(self, state, action, reward, next_state, done):
        # Add a new experience to the buffer
        self.buffer.append((state, action, reward, next_state, done))  # Append the experience tuple

    def sample(self, batch_size):
        # Sample a batch of experiences from the buffer
        transitions = random.sample(self.buffer, batch_size)  # Randomly sample experiences
        batch = list(zip(*transitions))  # Unzip the batch into separate lists
        state_batch = torch.cat(batch[0]).to(device)  # Concatenate and move to the correct device
        action_batch = torch.tensor(batch[1], dtype=torch.int64).unsqueeze(1).to(device)  # Action indices
        reward_batch = torch.tensor(batch[2], dtype=torch.float32).unsqueeze(1).to(device)  # Rewards
        next_state_batch = torch.cat(batch[3]).to(device)  # Next states
        done_batch = torch.tensor(batch[4], dtype=torch.float32).unsqueeze(1).to(device)  # Done flags
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch  # Return the batches

    def __len__(self):
        return len(self.buffer)  # Return the current size of the buffer


class Opponent:
    # Base class for different types of opponents
    def select_move(self, board):
        raise NotImplementedError  # Method to select move must be implemented by subclasses

"""

class OpeningBookOpponent(Opponent):
    def __init__(self, opening_book):
        self.book = chess.polyglot.MemoryMappedReader(opening_book)

    #def select_move(self, board):
    #    try:
    #        return self.book.choice(board).move
    #    except:
    #        return random.choice(list(board.legal_moves))


    def load_new_game(self):
        self.current_game = next(self.games)
        self.moves = list(self.current_game.mainline_moves())

    def select_move(self, board):
        # Return the master's move from the database
        return self.moves[board.fullmove_number - 1]
"""

class DQNAgent:
    def __init__(self, policy_net, action_size, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=50000):
        # Initialize the DQN agent with exploration parameters
        self.policy_net = policy_net  # Neural network that approximates the Q-function
        self.action_size = action_size  # Total number of possible actions
        self.epsilon_start = epsilon_start  # Starting value of epsilon (exploration rate)
        self.epsilon_end = epsilon_end  # Minimum value of epsilon
        self.epsilon_decay = epsilon_decay  # Rate at which epsilon decays
        self.epsilon = epsilon_start  # Initialize epsilon
        self.steps_done = 0  # Counter for the number of steps taken

    def select_action(self, state, board):
        # Select an action based on the current state and exploration rate
        sample = random.random()  # Generate a random number between 0 and 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.steps_done / self.epsilon_decay)  # Update epsilon
        self.steps_done += 1  # Increment the steps counter

        legal_moves = list(board.legal_moves)  # Get all legal moves in the current board state
        legal_indices = [encode_move(move) for move in legal_moves]  # Convert legal moves to indices

        if sample > self.epsilon:
            # Exploitation: select the best action according to the policy network
            with torch.no_grad():
                q_values = self.policy_net(state).squeeze(0)  # Get Q-values from the policy network
                masked_q_values = mask_legal_moves(board, q_values)  # Mask out illegal moves
                action_index = torch.argmax(masked_q_values).item()  # Select action with highest Q-value
        else:
            # Exploration: select a random legal action
            action_index = random.choice(legal_indices)

        return action_index  # Return the selected action index

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()  # Initialize the parent class
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # First convolutional layer
        self.bn1 = nn.BatchNorm2d(out_channels)  # Batch normalization after first conv layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)  # Second convolutional layer
        self.bn2 = nn.BatchNorm2d(out_channels)  # Batch normalization after second conv layer

        # Skip connection with 1x1 convolution if input and output channels differ
        self.skip = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),  # 1x1 convolution
            nn.BatchNorm2d(out_channels)  # Batch normalization
        ) if in_channels != out_channels else nn.Identity()  # Identity mapping if channels are the same

    def forward(self, x):
        identity = self.skip(x)  # Apply skip connection
        out = F.relu(self.bn1(self.conv1(x)))  # Apply first conv layer with ReLU activation
        out = self.bn2(self.conv2(out))  # Apply second conv layer
        return F.relu(out + identity)  # Add skip connection and apply ReLU activation
        #print(f"Position history length: {len(self.position_history)}")  # For debugging


class ChessLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ChessLSTM, self).__init__()  # Initialize the parent class
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)  # LSTM layer
        self.hidden_size = hidden_size  # Store hidden size for future use

    def forward(self, x, hidden=None):
        return self.lstm(x, hidden)  # Forward pass through the LSTM layer

class DeepChessDQNetwork(nn.Module):
    def __init__(self, action_size):
        super(DeepChessDQNetwork, self).__init__()  # Initialize the parent class

        # Input convolutional layer
        self.conv_input = nn.Conv2d(12, 64, kernel_size=3, padding=1)  # Input layer with 12 channels (pieces)
        self.bn_input = nn.BatchNorm2d(64)  # Batch normalization after input layer

        # Residual blocks
        self.res_blocks = nn.ModuleList([
            ResidualBlock(64, 64),    # First residual block
            ResidualBlock(64, 128),   # Second residual block
            ResidualBlock(128, 256),  # Third residual block
            ResidualBlock(256, 512)   # Fourth residual block
        ])

        # LSTM for position history
        self.pos_lstm = ChessLSTM(512 * 8 * 8, 1024)  # LSTM layer to process position history

        # Policy head (action selection)
        self.policy_conv = nn.Conv2d(512, 256, kernel_size=1)  # Convolutional layer for policy head
        self.policy_bn = nn.BatchNorm2d(256)  # Batch normalization for policy head
        self.policy_fc1 = nn.Linear(256 * 8 * 8, 2048)  # Fully connected layers for policy head
        self.policy_fc2 = nn.Linear(2048, 1024)
        self.policy_fc3 = nn.Linear(1024, action_size)  # Output layer matching the action space size

        # Value head (state evaluation)
        self.value_conv = nn.Conv2d(512, 128, kernel_size=1)  # Convolutional layer for value head
        self.value_bn = nn.BatchNorm2d(128)  # Batch normalization for value head
        self.value_fc1 = nn.Linear(128 * 8 * 8, 512)  # Fully connected layers for value head
        self.value_fc2 = nn.Linear(512, 256)
        self.value_fc3 = nn.Linear(256, 1)  # Output layer producing a single value

        # Position history
        self.position_history = []  # List to store previous positions
        self.max_history_length = 4  # Maximum number of positions to store

    def forward(self, x):
        # Forward pass through the network
        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))  # Apply input conv layer, batch norm, and ReLU
        x = x.contiguous()  # Ensure tensor is contiguous in memory

        # Fixed the residual blocks loop
        previous_x = x  # Store the input for skip connections
        for i, block in enumerate(self.res_blocks):
            x = block(x)  # Pass through the residual block
            x = x.contiguous()
            if i > 0:
                if previous_x.size(1) != x.size(1):
                    # Adjust previous_x if the number of channels has changed
                    previous_x = F.conv2d(
                        previous_x,
                        torch.ones(x.size(1), previous_x.size(1), 1, 1).to(x.device),
                        padding=0,
                        groups=1
                    )
                x = x + previous_x  # Add skip connection
            previous_x = x  # Update previous_x

        # Store current position
        batch_size = x.size(0)  # Get the batch size
        current_position = x.contiguous().view(batch_size, -1)  # Flatten the feature maps

        # Handle position history
        self.position_history.append(current_position.detach())  # Add current position to history
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)  # Remove oldest position if history is too long
        #print(f"Position history length: {len(self.position_history)}")  # For debugging

        # LSTM processing if history exists
        if len(self.position_history) > 1:
            try:
                history_tensor = torch.stack(self.position_history, dim=1)  # Stack positions into a sequence
                lstm_out, _ = self.pos_lstm(history_tensor)  # Pass through LSTM
                lstm_features = lstm_out[:, -1, :]  # Get the output corresponding to the last position
                # Reshape lstm_features correctly
                lstm_features = lstm_features.view(batch_size, 512, 8, 8).contiguous()
                x = x + lstm_features  # Add LSTM features to the current features
            except RuntimeError as e:
                # Just continue without LSTM features if there's an error
                pass

        # Policy head
        policy = self.policy_conv(x)  # Apply convolutional layer
        policy = self.policy_bn(policy)  # Apply batch normalization
        policy = F.relu(policy)  # Apply ReLU activation
        policy = policy.contiguous().view(batch_size, -1)  # Flatten the feature maps
        policy = F.relu(self.policy_fc1(policy))  # Apply fully connected layers with ReLU
        policy = F.relu(self.policy_fc2(policy))
        policy = self.policy_fc3(policy)  # Output layer for policy head

        # Value head
        value = self.value_conv(x)  # Apply convolutional layer
        value = self.value_bn(value)  # Apply batch normalization
        value = F.relu(value)  # Apply ReLU activation
        value = value.contiguous().view(batch_size, -1)  # Flatten the feature maps
        value = F.relu(self.value_fc1(value))  # Apply fully connected layers with ReLU
        value = F.relu(self.value_fc2(value))
        value = torch.tanh(self.value_fc3(value))  # Output layer with tanh activation for value head

        # Combine policy and value
        return policy + value.expand(-1, policy.size(1))  # Combine policy and value outputs

    def reset_history(self):
        self.position_history = []  # Reset the position history

# Add this line: Define a RandomOpponent class
class RandomOpponent(Opponent):
    # Opponent that selects moves randomly
    def select_move(self, board):
        # Return a random legal move
        return random.choice(list(board.legal_moves))

# Add this line: Define an AgentOpponent class
class AgentOpponent(Opponent):
    def __init__(self, agent):
        self.agent = agent  # Reference to the agent's policy network

    def select_move(self, board):
        # Get the device from the agent's policy_net parameters
        device = next(self.agent.policy_net.parameters()).device
        # Encode the board state
        state = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0)\
                .permute(0, 3, 1, 2).to(device)
        # Agent selects an action
        action_index = self.agent.select_action(state, board)
        # Decode the move
        move = decode_move(action_index)
        # Handle illegal moves
        if move not in board.legal_moves:
            move = random.choice(list(board.legal_moves))
        # Return the selected move
        return move

# Add this line: Define a StockfishOpponent class
class StockfishOpponent(Opponent):
    def __init__(self, path_to_engine, depth=1):
        #self.engine = chess.engine.SimpleEngine.popen_uci(path_to_engine)  # Open the Stockfish engine
        self.path_to_engine = path_to_engine

        self.depth = depth  # Set search depth

    def select_move(self, board):
        print(f"Stockfish (depth {self.depth}) is about to select a move...")
        with chess.engine.SimpleEngine.popen_uci(self.path_to_engine) as engine:
            result = engine.play(board, chess.engine.Limit(depth=self.depth))
        print("Stockfish has selected a move.")
        return result.move
        #print(f"Stockfish (depth {self.depth}) is about to select a move...")
        # Get move from Stockfish
        #result = self.engine.play(board, chess.engine.Limit(depth=self.depth))
        # Return the move
        #print("Stockfish has selected a move.")
        #return result.move

    def close(self):
        pass
        print("Attempting to close Stockfish engine...")
        #self.engine.close()  # Unconditionally close the engine
        print("Stockfish engine closed.")

        #self.engine.quit()  # Method to close the engine

class SelfPlayManager:
    def __init__(self, agent, opponent, device=device):
        self.agent = agent  # The agent being trained
        self.opponent = opponent  # The opponent the agent is playing against
        self.device = device  # Device to run computations on

    def play_game(self):
        # Reset the position history for the agent's policy network
        self.agent.policy_net.reset_history()

        # If the opponent is also using a network with position history, reset it
        if isinstance(self.opponent, AgentOpponent):
            self.opponent.agent.policy_net.reset_history()

        board = chess.Board()  # Initialize a new chess board
        experiences = []  # List to store experiences (state, action, reward, next_state, done)
        max_moves = 75  # Maximum number of moves to prevent infinite games
        move_count = 0  # Counter for the number of moves made

        while not board.is_game_over() and move_count < max_moves:
            move_count += 1  # Increment move count
            state = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0) \
                .permute(0, 3, 1, 2).to(self.device)  # Encode the current board state

            if board.turn == chess.WHITE:
                # Agent's turn (assuming agent plays white)
                action_index = self.agent.select_action(state, board)  # Agent selects an action
                move = decode_move(action_index)  # Decode the action into a move
                if move not in board.legal_moves:
                    # If the move is illegal, select a random legal move
                    move = random.choice(list(board.legal_moves))
                    action_index = encode_move(move)
            else:
                # Opponent's turn
                move = self.opponent.select_move(board)  # Opponent selects a move
                action_index = encode_move(move)

            # Push the move onto the board
            board.push(move)
            # Re-encode the action index
            action_index = encode_move(move)
            # Determine if it's agent's turn (after the move)
            is_agent_turn = board.turn != chess.WHITE  # Since the agent just moved

            # Calculate the reward
            reward = get_reward(board, move, board.is_game_over(), is_agent_turn)

            if board.is_game_over():
                # Append terminal experience
                experiences.append((state, action_index, reward, None, True))
            else:
                # Get next state
                next_state = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0) \
                    .permute(0, 3, 1, 2).to(self.device)
                # Append experience
                experiences.append((state, action_index, reward, next_state, False))

        print(f"\nGame finished with result: {board.result()}")  # Output the result of the game

        return experiences, board.result()  # Return the experiences and the game result
def load_master_games(pgn_database_path):
    games = []
    with open(pgn_database_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            games.append(game)
    return games
def extract_state_action_pairs(games):
    state_action_pairs = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            # Encode the current board state
            state = torch.tensor(encode_board(board), dtype=torch.float32).unsqueeze(0) \
                .permute(0, 3, 1, 2).to(device)
            # Encode the move
            action_index = encode_move(move)
            state_action_pairs.append((state, action_index))
            board.push(move)
    return state_action_pairs


def train_double_dqn(agent, target_net, optimizer, replay_buffer, batch_size, gamma):
    # Function to train the agent using Double DQN algorithm
    if len(replay_buffer) < batch_size:
        return None  # Not enough samples to train

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(batch_size)
    # Sample a batch of experiences from the replay buffer

    current_q_values = agent.policy_net(state_batch).gather(1, action_batch)
    # Get the Q-values for the actions taken

    with torch.no_grad():
        # Use policy_net to SELECT action (argmax)
        next_state_actions = agent.policy_net(next_state_batch).max(1)[1].unsqueeze(1)

        #next_q_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
        # Get the maximum Q-values for the next states from the target network
        #expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))
        # Compute the expected Q-values using the Bellman equation
        # Use target_net to EVALUATE action
        next_q_values = target_net(next_state_batch).gather(1, next_state_actions)
        expected_q_values = reward_batch + (gamma * next_q_values * (1 - done_batch))

    loss = F.smooth_l1_loss(current_q_values, expected_q_values)
    # Compute the loss between current and expected Q-values

    optimizer.zero_grad()  # Zero the gradients
    loss.backward()  # Backpropagate the loss
    torch.nn.utils.clip_grad_norm_(agent.policy_net.parameters(), max_norm=1.0)
    # Clip gradients to prevent exploding gradients
    optimizer.step()  # Update the network parameters

    return loss.item()  # Return the loss value

if __name__ == "__main__":

    #num_episodes = 9000  # Start with a small number for testing
    gamma = 0.99  # Discount factor for future rewards
    batch_size = 128  # Batch size for training
    target_update = 3  # Frequency of updating the target network
    memory_capacity = 20000  # Capacity of the replay buffer
    action_size = 64 * 64 + 4096 * 4  # Including promotions
    print("Initializing training...")
    print(f"Action space size: {64 * 64 + 4096 * 4}")
    print(f"Batch size: {batch_size}")
    print(f"Memory capacity: {memory_capacity}")
    print(f"Device: {device}")
    print(f"Network architecture:")
    print(f" - Input: 12 channels (board state)")
    print(f" - Residual blocks: 64->64->128->256->512")
    print(f" - LSTM memory: {4} positions")
    print(f" - Output: {action_size} actions")
    print("-" * 50)

    # Initialize policy_net and target_net
    policy_net = DeepChessDQNetwork(action_size).to(device)  # Main network for action selection
    target_net = DeepChessDQNetwork(action_size).to(device)  # Target network for stability

    # Load from episode 9000 checkpoint
    print("Loading checkpoint from episode 9000...")
    checkpoint_path = 'checkpoint_ep9000.pth'
    checkpoint = torch.load(checkpoint_path)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    target_net.load_state_dict(checkpoint['policy_net_state_dict'])  # Make sure target net is synced
    episode = 9000 #explicit call to prior run of terminated training final stable checkpoint

    target_net.load_state_dict(policy_net.state_dict())  # Initialize target network with policy network weights
    target_net.eval()  # Set target network to evaluation mode

    # Initialize optimizer and scheduler
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)  # Adam optimizer with learning rate 0.0001
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=50,
        verbose=True
    )  # Learning rate scheduler to reduce LR when loss plateaus

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    # Start counting episodes from where we left off

    print(f"Successfully loaded checkpoint from episode {episode}")


    # Initialize replay buffer and agent
    replay_buffer = ReplayBuffer(memory_capacity)  # Experience replay buffer
    agent = DQNAgent(
        policy_net,
        action_size,
        epsilon_start=0.852,  # Continue from episode 9000's epsilon
        epsilon_end=0.1,
        epsilon_decay=10000
    )
    agent.steps_done = 9000  # Also set the steps

    # Training statistics
    stats = {
        'wins': 0,
        'draws': 0,
        'losses': 0,
        'avg_game_length': []
    }

    # Define the curriculum
    curriculum = [
        #{'type': 'opening_book', 'episodes': 50, 'book_path': OPENING_BOOK_PATH},  # Learn standard openings
        {'type': 'self_play', 'episodes': 1000},
        {'type': 'master_games', 'episodes': 10, 'database_path': MASTER_GAMES_PATH},  # Learn from masters
        {'type': 'stockfish', 'depth': 1, 'episodes': 100},
        {'type': 'stockfish', 'depth': 2, 'episodes': 200},
        {'type': 'stockfish', 'depth': 3, 'episodes': 300},
        {'type': 'stockfish', 'depth': 4, 'episodes': 300},
        {'type': 'stockfish', 'depth': 5, 'episodes': 300},
        {'type': 'stockfish', 'depth': 10, 'episodes': 500},
        {'type': 'stockfish', 'depth': 15, 'episodes': 500},
        {'type': 'stockfish', 'depth': 20, 'episodes': 500},
        {'type': 'stockfish', 'depth': 25, 'episodes': 500},
        {'type': 'stockfish', 'depth': 30, 'episodes': 500},
        {'type': 'stockfish', 'depth': 35, 'episodes': 500},
        {'type': 'stockfish', 'depth': 40, 'episodes': 500},
        # Add more stages as needed
    ]

    total_episodes = sum(stage['episodes'] for stage in curriculum)  # Total number of episodes
    opponent_index = 0  # Index of the current opponent in the curriculum
    opponent_episode_start = 0  # Episode number when the current opponent started

"""
    # Initialize the first opponent based on the curriculum
    stage = curriculum[opponent_index]
    if stage['type'] == 'self_play':
        opponent = AgentOpponent(agent)
    elif stage['type'] == 'random':
        opponent = RandomOpponent()
    elif stage['type'] == 'stockfish':
        opponent = StockfishOpponent(path_to_engine=STOCKFISH_PATH, depth=stage['depth'])
    elif stage['type'] == 'opening_book':
        opponent = OpeningBookOpponent(stage['book_path'])
    elif stage['type'] == 'master_games':
        opponent = MasterGamesOpponent(stage['database_path'])
    else:
        raise ValueError(f"Unknown opponent type: {stage['type']}")
"""
episode = 0
opponent = None
self_play_manager = None
opponent_index = 0
opponent_episode_start = 0
# Initialize the first stage
stage = curriculum[opponent_index]

# Initialize the first opponent if the first stage is not 'master_games'
if stage['type'] == 'master_games':
    opponent = None  # Will be handled inside the loop
else:
    if stage['type'] == 'self_play':
        opponent = AgentOpponent(agent)
    elif stage['type'] == 'random':
        opponent = RandomOpponent()
    elif stage['type'] == 'stockfish':
        opponent = StockfishOpponent(path_to_engine='/opt/homebrew/bin/stockfish', depth=stage['depth'])
    else:
        raise ValueError(f"Unknown opponent type: {stage['type']}")

# If opponent is initialized, create the SelfPlayManager
if opponent is not None:
    self_play_manager = SelfPlayManager(agent, opponent)

# Start the training loop
while episode < total_episodes:
    # Check if we need to switch to the next stage
    if episode - opponent_episode_start >= stage['episodes']:
        # Close previous opponent if needed
        if opponent is not None and isinstance(opponent, StockfishOpponent):
            print("Closing the Stockfish opponent...")
            opponent.close()
            print("Stockfish opponent closed.")

        # Move to the next stage
        opponent_index += 1
        opponent_episode_start = episode
        # Check if we've completed the curriculum
        if opponent_index >= len(curriculum):
            break  # Finished all stages

        # Initialize the new stage
        stage = curriculum[opponent_index]


        # Handle the 'master_games' stage
        if stage['type'] == 'master_games':
            #continue

            # Perform supervised learning
            print(f"Starting supervised learning from master games for {stage['episodes']} episodes.")
            # Load master games
            #games = load_master_games(stage['database_path'])
            games = load_master_games(stage['database_path'])[:500]  # Load only first 2 games
            print(f"Loaded {len(games)} master games")  # Debug print 1

            print("Starting to extract state-action pairs...")  # New debug print

            # Extract state-action pairs
            state_action_pairs = extract_state_action_pairs(games)

            print(f"Extracted {len(state_action_pairs)} state-action pairs")  # Debug print 2

            # Shuffle the data
            random.shuffle(state_action_pairs)
            # Create a DataLoader for batching
            states = torch.cat([s for s, a in state_action_pairs], dim=0)
            print(f"States tensor created with shape: {states.shape}")  # New debug print

            actions = torch.tensor([a for s, a in state_action_pairs], dtype=torch.long).to(device)
            print(f"Actions tensor created with shape: {actions.shape}")  # New debug print

            print(f"Created tensors with {len(states)} states")  # Debug print 3
            print("Creating dataset and dataloader...")  # New debug print

            dataset = TensorDataset(states, actions)
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            print("DataLoader created successfully")  # New debug print

            # Supervised learning
            for sup_episode in range(stage['episodes']):
                total_loss = 0.0
                for state_batch, action_batch in data_loader:
                    optimizer.zero_grad()
                    q_values = agent.policy_net(state_batch)
                    # Use cross-entropy loss
                    loss = F.cross_entropy(q_values, action_batch)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(data_loader)
                # Log the average loss
                writer.add_scalar('Loss/supervised', avg_loss, episode + sup_episode)
                print(f"Supervised Episode {episode + sup_episode}, Loss: {avg_loss:.4f}")
                # After supervised learning, increment episode and continue
            episode += stage['episodes']
            opponent = None  # No opponent during supervised learning
            self_play_manager = None  # Ensure self_play_manager is None
            continue  # Skip the rest of the loop to avoid playing games during this stage

    # For other stages, initialize the opponent as usual
    if stage['type'] == 'self_play':
        opponent = AgentOpponent(agent)
    elif stage['type'] == 'random':
        opponent = RandomOpponent()
    elif stage['type'] == 'stockfish':
        opponent = StockfishOpponent(path_to_engine='/opt/homebrew/bin/stockfish', depth=stage['depth'])
    else:
        raise ValueError(f"Unknown opponent type: {stage['type']}")

    # Initialize SelfPlayManager with the opponent
    self_play_manager = SelfPlayManager(agent, opponent)


    # Proceed with training if not in 'master_games' stage
    if stage['type'] != 'master_games':
        current_time = time.strftime('%H:%M:%S', time.localtime())
        print(f"\n[{current_time}] Starting episode {episode}")
        experiences, result = self_play_manager.play_game()  # Play a game and collect experiences

        for exp in experiences:
            if exp[3] is not None:
                replay_buffer.push(*exp)  # Add experiences to replay buffer

            # Update statistics
        stats['avg_game_length'].append(len(experiences))  # Record game length
        if result == '1-0':
            stats['wins'] += 1  # Agent won
        elif result == '0-1':
            stats['losses'] += 1  # Agent lost
        else:
            stats['draws'] += 1  # Game was a draw

        current_time = time.strftime('%H:%M:%S', time.localtime())
        print(f"[{current_time}] Episode {episode}")
        print(f"Game length: {len(experiences)} moves")
        print(f"Result: {result}")
        print(f"Win rate: {stats['wins'] / (episode + 1):.2%}")
        print(f"Average game length: {sum(stats['avg_game_length']) / len(stats['avg_game_length']):.1f}")

        # Training step
        loss = train_double_dqn(agent, target_net, optimizer, replay_buffer, batch_size, gamma)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())  # Update the target network

        if loss is not None:
            # Adjust learning rate with scheduler
            scheduler.step(loss)
            # Log metrics
            writer.add_scalar('Loss/train', loss, episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Draws', stats['draws'], episode)
            writer.add_scalar('Losses', stats['losses'], episode)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], episode)
            print(f"Episode {episode}, Loss: {loss:.4f}, Epsilon: {agent.epsilon:.4f}, Result: {result}")

        if episode % 500 == 0:
            torch.save({
                'episode': episode,
                'policy_net_state_dict': policy_net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, f'checkpoint_ep{episode}.pth')  # Save model checkpoint

        # Delete older checkpoint (keep only last few)
        old_checkpoint = f'checkpoint_ep{episode - 1500}.pth'  # Delete checkpoints older than 3 saves ago
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)  # Remove the old checkpoint

        episode += 1  # Increment episode counter
print("Training completed.")
# Flush and close the TensorBoard writer
writer.flush()
writer.close()
print("TensorBoard writer flushed and closed.")

print("TensorBoard writer closed.")
torch.save(policy_net.state_dict(), 'final_model.pth')  # Save the final model
print("Final model saved.")
import threading
import multiprocessing

print("Active threads:")
for thread in threading.enumerate():
    print(f"- {thread.name}")

print("Active child processes:")
for process in multiprocessing.active_children():
    print(f"- PID: {process.pid}, Name: {process.name}")