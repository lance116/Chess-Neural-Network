import pygame
import sys
import random
import os
import copy
import math
import time
import traceback
import numpy as np
import chess
import chess.pgn
import json

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("TensorFlow using GPU:", physical_devices[0])
    else:
        print("TensorFlow using CPU.")
except ImportError:
    print("TensorFlow not found.")
    sys.exit()
except Exception as e:
    print(f"TensorFlow GPU check failed (continuing with CPU): {e}")


WIDTH, HEIGHT = 600, 640
BOARD_SIZE = 8
SQUARE_SIZE = WIDTH // BOARD_SIZE
CAPTURED_AREA_HEIGHT = 40
GAME_AREA_HEIGHT = HEIGHT - CAPTURED_AREA_HEIGHT
FPS = 30
AI_THINK_TIME = 5.0
ANIMATION_DURATION = 6

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (238, 238, 210)
DARK_SQUARE = (118, 150, 86)
HIGHLIGHT_COLOR = (255, 255, 0, 150)
POSSIBLE_MOVE_COLOR = (135, 206, 250, 150)
CHECK_HIGHLIGHT_COLOR = (255, 0, 0, 100)
MENU_BG_COLOR = (40, 40, 40)
MENU_TEXT_COLOR = (220, 220, 220)
CAPTURED_BG_COLOR = (60, 60, 60)
CAPTURED_TEXT_COLOR = (200, 200, 200)
FORFEIT_BUTTON_COLOR = (180, 0, 0)
FORFEIT_TEXT_COLOR = (255, 255, 255)
PROMOTION_BG_COLOR = (50, 50, 50, 230)
PROMOTION_BORDER_COLOR = (150, 150, 150)
PROMOTION_CHOICE_COLOR = (80, 80, 80)
PROMOTION_TEXT_COLOR = (220, 220, 220)

PIECES = {
    'wP', 'wR', 'wN', 'wB', 'wQ', 'wK',
    'bP', 'bR', 'bN', 'bB', 'bQ', 'bK'
}
IMAGE_PATH = 'images'

PIECE_VALUES = {
    'P': 100, 'N': 300, 'B': 300, 'R': 500, 'Q': 900, 'K': 10000
}
POSITION_SCORES = {
    'P': [ [0,  0,  0,  0,  0,  0,  0,  0], [90, 90, 90, 90, 90, 90, 90, 90], [30, 30, 40, 60, 60, 40, 30, 30], [10, 10, 20, 40, 40, 20, 10, 10], [5,  5, 10, 20, 20, 10,  5,  5], [0,  0,  0,-10,-10,  0,  0,  0], [5, -5,-10,  0,  0,-10, -5,  5], [0,  0,  0,  0,  0,  0,  0,  0] ],
    'N': [ [-50,-40,-30,-30,-30,-30,-40,-50], [-40,-20,  0,  5,  5,  0,-20,-40], [-30,  5, 10, 15, 15, 10,  5,-30], [-30,  5, 15, 20, 20, 15,  5,-30], [-30,  5, 15, 20, 20, 15,  5,-30], [-30,  5, 10, 15, 15, 10,  5,-30], [-40,-20,  0,  0,  0,  0,-20,-40], [-50,-40,-30,-30,-30,-30,-40,-50] ],
    'B': [ [-20,-10,-10,-10,-10,-10,-10,-20], [-10,  0,  0,  0,  0,  0,  0,-10], [-10,  0,  5, 10, 10,  5,  0,-10], [-10,  5,  5, 10, 10,  5,  5,-10], [-10,  0, 10, 15, 15, 10,  0,-10], [-10, 10, 10, 10, 10, 10, 10,-10], [-10,  5,  0,  0,  0,  0,  5,-10], [-20,-10,-10,-10,-10,-10,-10,-20] ],
    'R': [ [ 0,  0,  0,  0,  0,  0,  0,  0], [ 5, 20, 20, 20, 20, 20, 20,  5], [-5,  0,  0,  0,  0,  0,  0, -5], [-5,  0,  0,  0,  0,  0,  0, -5], [-5,  0,  0,  0,  0,  0,  0, -5], [-5,  0,  0,  0,  0,  0,  0, -5], [-5,  0,  0,  0,  0,  0,  0, -5], [ 0,  0,  0,  5,  5,  0,  0,  0] ],
    'Q': [ [-20,-10,-10, -5, -5,-10,-10,-20], [-10,  0,  0,  0,  0,  0,  0,-10], [-10,  0,  5,  5,  5,  5,  0,-10], [ -5,  0,  5, 10, 10,  5,  0, -5], [ -5,  0,  5, 10, 10,  5,  0, -5], [-10,  0,  5,  5,  5,  5,  0,-10], [-10,  0,  0,  0,  0,  0,  0,-10], [-20,-10,-10, -5, -5,-10,-10,-20] ],
    'K': [ [-50,-30,-30,-30,-30,-30,-30,-50], [-30,-30,  0,  0,  0,  0,-30,-30], [-30,-10, 20, 30, 30, 20,-10,-30], [-30,-10, 30, 40, 40, 30,-10,-30], [-30,-10, 30, 40, 40, 30,-10,-30], [-30,-10, 20, 30, 30, 20,-10,-30], [-30,-20,-10,  0,  0,-10,-20,-30], [-50,-40,-30,-20,-20,-30,-40,-50] ]
}

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Chess Game")
clock = pygame.time.Clock()
font_menu_title = pygame.font.SysFont(None, 70)
font_menu_option = pygame.font.SysFont(None, 50)
font_status = pygame.font.SysFont(None, 28)
font_captured = pygame.font.SysFont(None, 22)
font_game_over = pygame.font.SysFont(None, 60)
font_restart = pygame.font.SysFont(None, 30)
font_button = pygame.font.SysFont(None, 26)
font_promotion = pygame.font.SysFont(None, 40)

piece_images = {}

try:
    missing_files = []
    for piece in PIECES:
        path = os.path.join(IMAGE_PATH, f"{piece}.png")
        try:
            img = pygame.image.load(path).convert_alpha()
            piece_images[piece] = pygame.transform.scale(img, (SQUARE_SIZE, SQUARE_SIZE))
        except pygame.error as e:
            if "Unable to open file" in str(e):
                missing_files.append(path)
            else:
                print(f"Error loading or scaling image {path}: {e}")
        except FileNotFoundError:
             missing_files.append(path)

    if missing_files:
        print("Error: The following image files are missing or could not be loaded:")
        for f in missing_files:
            print(f"- {f}")
        print("Please ensure all piece images exist in the 'images' folder.")
        sys.exit()

    if len(piece_images) != len(PIECES):
         print("Error: Not all piece images were loaded successfully.")
         sys.exit()

except Exception as e:
    print(f"An unexpected error occurred during image loading: {e}")
    sys.exit()


initial_board_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

board = None
selected_piece = None
legal_moves = []
current_turn = 'w'
game_over = False
winner = None
status_message = ""
# castling_rights managed by python-chess board object
game_state = 'MENU'
player_color = None
move_history = []

is_animating = False
anim_piece_img = None
anim_start_pos_screen = (0, 0)
anim_end_pos_screen = (0, 0)
anim_current_pos_screen = (0, 0)
anim_progress = 0
pending_move = None

promotion_pos = None
promotion_color = None
promotion_choices = ['Q', 'R', 'B', 'N']
promotion_choice_rects = []

forfeit_button_rect = pygame.Rect(WIDTH // 2 - 40, GAME_AREA_HEIGHT + 5, 80, CAPTURED_AREA_HEIGHT - 10)

piece_symbols = [None, 'P', 'N', 'B', 'R', 'Q', 'K']
piece_map = {symbol: i for i, symbol in enumerate(piece_symbols)}

N_PIECE_TYPES = 6
N_COLORS = 2
N_PLANES = N_PIECE_TYPES * N_COLORS + 1 # 12 piece planes + 1 plane for whose turn
INPUT_SHAPE = (BOARD_SIZE, BOARD_SIZE, N_PLANES)
MODEL_WEIGHTS_FILE = 'chess_model.weights.h5'
TRAINING_PROGRESS_FILE = 'training_progress.json'
PGN_DATABASE_PATH = r"A:\\Chess Neural Network\\Lichess Elite Database\\Lichess Elite Database"
TRAIN_MODEL_ON_START = False # Flag to run model training on startup. Set to False after initial training.
TRAINING_GAME_LIMIT = 1000 # Increased from 200 to get more training data

def build_model(input_shape):
    # Create a balanced model with stronger regularization to prevent overfitting
    from tensorflow.keras import regularizers
    
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            # First convolutional block with more filters
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same",
                          kernel_regularizer=regularizers.l2(0.003)),  # Increased L2 regularization
            layers.BatchNormalization(),
            # Second convolutional block
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same",
                          kernel_regularizer=regularizers.l2(0.003)),
            layers.BatchNormalization(),
            # Feature processing
            layers.Flatten(),
            layers.Dense(192, activation="relu", kernel_regularizer=regularizers.l2(0.003)),  # Reduced size
            layers.Dropout(0.4),  # Increased dropout
            layers.Dense(96, activation="relu", kernel_regularizer=regularizers.l2(0.003)),   # Reduced size
            layers.Dropout(0.4),  # Increased dropout
            layers.Dense(1, activation="sigmoid")  # Output [0, 1] for win probability for white
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model(INPUT_SHAPE)
model.summary()

if os.path.exists(MODEL_WEIGHTS_FILE):
    try:
        # Backup weights before trying to load them, in case of shape mismatch
        if os.path.getsize(MODEL_WEIGHTS_FILE) > 0:
            backup_file = f"{MODEL_WEIGHTS_FILE}.backup"
            import shutil
            shutil.copy2(MODEL_WEIGHTS_FILE, backup_file)
            print(f"Created backup of weights at {backup_file}")
        
        model.load_weights(MODEL_WEIGHTS_FILE)
        print(f"Loaded model weights from {MODEL_WEIGHTS_FILE}")
    except ValueError as ve:
        if "shape mismatch" in str(ve) or "incompatible" in str(ve):
            print(f"Error: Model architecture changed, weights are incompatible: {ve}")
            print("Weights backup was created. Consider reverting to previous model architecture.")
            # Could add automatic restoration of architecture here if we have it stored
            print("Starting with initial weights.")
        else:
            print(f"Error loading weights from {MODEL_WEIGHTS_FILE}: {ve}. Starting with initial weights.")
    except Exception as e:
        print(f"Error loading weights from {MODEL_WEIGHTS_FILE}: {e}. Starting with initial weights.")
else:
    print(f"Weights file {MODEL_WEIGHTS_FILE} not found. Starting with initial weights.")

def board_to_bitboard(board_pychess):
    bitboard = np.zeros(INPUT_SHAPE, dtype=np.float32)
    plane_index = 0
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]:
            for square in board_pychess.pieces(piece_type, color):
                row = 7 - (square // 8)
                col = square % 8
                bitboard[row, col, plane_index] = 1
            plane_index += 1

    bitboard[:, :, 12] = 1.0 if board_pychess.turn == chess.WHITE else 0.0

    return np.expand_dims(bitboard, axis=0)


def load_parse_and_train(pgn_dir_path, model_to_train, weights_save_path, game_limit=None):
    print(f"Starting PGN parsing from: {pgn_dir_path}")
    X_train = []
    y_train = []
    games_parsed = 0
    positions_parsed = 0
    
    # Load training progress if it exists
    processed_files = set()
    total_games_processed = 0
    current_file_position = 0  # Position in the current file where we left off
    current_file_name = None   # File we were processing last time
    
    if os.path.exists(TRAINING_PROGRESS_FILE):
        try:
            with open(TRAINING_PROGRESS_FILE, 'r') as f:
                training_progress = json.load(f)
                processed_files = set(training_progress.get('processed_files', []))
                total_games_processed = training_progress.get('total_games_processed', 0)
                current_file_name = training_progress.get('current_file_name', None)
                current_file_position = training_progress.get('current_file_position', 0)
                print(f"Loaded training progress: {len(processed_files)} files fully processed, {total_games_processed} games processed in total")
                if current_file_name and current_file_position > 0:
                    print(f"Resuming from file {current_file_name} at position {current_file_position}")
        except Exception as e:
            print(f"Error loading training progress: {e}. Starting fresh.")
    
    pgn_files = [os.path.join(pgn_dir_path, f) for f in os.listdir(pgn_dir_path) if f.lower().endswith('.pgn')]
    if not pgn_files:
        print(f"Error: No .pgn files found in directory: {pgn_dir_path}")
        return
        
    # Sort files alphabetically for consistency between runs
    pgn_files.sort()
    
    # Filter out fully processed files
    pgn_files = [f for f in pgn_files if os.path.basename(f) not in processed_files]
    
    # If we have a current file that was partially processed, make sure it's first in the list
    if current_file_name and current_file_position > 0:
        current_file_path = os.path.join(pgn_dir_path, current_file_name)
        if current_file_path in pgn_files:
            # Move current_file to the beginning of the list
            pgn_files.remove(current_file_path)
            pgn_files.insert(0, current_file_path)
    
    print(f"Found {len(pgn_files)} PGN files that need processing.")
    
    for pgn_file_path in pgn_files:
        if game_limit is not None and games_parsed >= game_limit:
            break
        
        file_basename = os.path.basename(pgn_file_path)
        is_resuming = (file_basename == current_file_name)
        start_pos = current_file_position if is_resuming else 0
        
        print(f"Processing file: {file_basename}{' (resuming)' if is_resuming else ''}...")
        
        try:
            with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                # Skip to the saved position if resuming
                if start_pos > 0:
                    pgn_file.seek(start_pos)
                    print(f"Skipped to position {start_pos} in file")
                
                while True:
                    # Remember position before reading each game
                    current_position = pgn_file.tell()
                    
                    if game_limit is not None and games_parsed >= game_limit:
                        # Remember where we stopped
                        current_file_position = current_position
                        current_file_name = file_basename
                        break
                        
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        # We've reached the end of this file
                        current_file_position = 0  # Reset for next file
                        current_file_name = None
                        break

                    games_parsed += 1
                    try:
                        result = game.headers["Result"]
                        if result == '1-0':
                            outcome = 1.0 # White wins
                        elif result == '0-1':
                            outcome = 0.0 # Black wins (from white's perspective)
                        elif result == '1/2-1/2':
                            outcome = 0.5 # Draw
                        else:
                            continue # Skip unknown results

                        board_iter = game.board()
                        for move in game.mainline_moves():
                            board_iter.push(move)
                            # Store position and eventual outcome
                            bitboard_tensor = board_to_bitboard(board_iter)[0] # Remove batch dim for list
                            X_train.append(bitboard_tensor)
                            y_train.append(outcome)
                            positions_parsed += 1

                        if games_parsed % 100 == 0:
                            print(f" Parsed {games_parsed} games, {positions_parsed} positions...")

                    except KeyError:
                        print(f"Skipping game {games_parsed}: Missing 'Result' header.")
                    except Exception as e_inner:
                        print(f"Error processing game {games_parsed}: {e_inner}")
                        traceback.print_exc()

        except FileNotFoundError:
            print(f"Error: PGN file not found at {pgn_file_path}")
        except Exception as e_outer:
            print(f"Error reading PGN file {pgn_file_path}: {e_outer}")
            traceback.print_exc()

    if not X_train:
        print("No valid positions parsed from PGN files. Training cannot proceed.")
        return
        
    print(f"\nTotal games parsed: {games_parsed}")
    print(f"Total positions extracted: {len(X_train)}")
    
    X_train_np = np.array(X_train, dtype=np.float32)
    y_train_np = np.array(y_train, dtype=np.float32)
    
    # Before training, save the progress of processed files
    # Get the list of fully processed files (all files except the last one we were working on)
    files_processed_this_run = []
    for pgn_path in pgn_files:
        filename = os.path.basename(pgn_path)
        if filename not in processed_files:
            files_processed_this_run.append(filename)
    
    # Don't include the last file if we're not done with it
    if files_processed_this_run and game_limit is not None and games_parsed >= game_limit:
        # We hit the limit during processing
        last_processed_file = files_processed_this_run[-1]
        files_fully_processed = files_processed_this_run[:-1]
        current_file_name = last_processed_file
    else:
        # We processed all files or didn't hit the limit
        files_fully_processed = files_processed_this_run
        current_file_name = None

    # Update the processed files set
    processed_files.update(files_fully_processed)
    
    # Save progress - now with the correct file position
    training_progress = {
        'processed_files': list(processed_files),
        'total_games_processed': total_games_processed + games_parsed,
        'current_file_name': current_file_name,
        'current_file_position': current_file_position
    }
    
    try:
        with open(TRAINING_PROGRESS_FILE, 'w') as f:
            json.dump(training_progress, f)
        print(f"Saved training progress: {len(processed_files)} files processed")
        if current_file_name:
            print(f"Next run will resume from file {current_file_name} at position {current_file_position}")
    except Exception as e:
        print(f"Warning: Could not save training progress: {e}")

    print("Starting model training...")
    try:
        # Add early stopping to prevent overfitting
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )        # Add model checkpoint to save best model
        model_checkpoint = keras.callbacks.ModelCheckpoint(
            weights_save_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        
        # Apply label noise to reduce overfitting (common technique to induce underfitting)
        noise_factor = 0.08  # Increased from 0.05 to reduce overfitting
        noisy_labels = y_train_np.copy()
        indices = np.random.choice(len(y_train_np), size=int(len(y_train_np) * noise_factor), replace=False)
        for i in indices:
            noisy_labels[i] = np.random.random() # Random value between 0-1
        
        history = model_to_train.fit(
            X_train_np,
            noisy_labels,  # Use noisy labels to induce underfitting
            epochs=5,  # Increased epochs
            batch_size=64,  # Increased batch size for better gradient estimates
            validation_split=0.1,
            shuffle=True,
            verbose=1,
            callbacks=[early_stopping, model_checkpoint]
        )
        print("Training complete.")
        # Weights are already saved by ModelCheckpoint
    except Exception as e_train:
        print(f"Error during model training: {e_train}")
        traceback.print_exc()


if TRAIN_MODEL_ON_START:
    load_parse_and_train(PGN_DATABASE_PATH, model, MODEL_WEIGHTS_FILE, game_limit=TRAINING_GAME_LIMIT)
    print("Training finished. Set TRAIN_MODEL_ON_START to False for next run.")
    # Uncomment to exit after training, if you only want to train the model.
    # sys.exit()

def evaluate_board_nn(board_pychess):
    if board_pychess.is_insufficient_material() or board_pychess.is_stalemate():
        return 0.0 # Draw evaluation
    if board_pychess.is_checkmate():
        # Return extreme value favoring the winner
        return -10000.0 if board_pychess.turn == chess.WHITE else 10000.0

    board_tensor = board_to_bitboard(board_pychess)
    predicted_value = model.predict(board_tensor, verbose=0)[0][0] # Output is [0, 1] prob of white win

    # Convert probability [0, 1] to a score roughly in centipawns for minimax
    # 0.5 -> 0, 1.0 -> +High, 0.0 -> -High
    # Example: map to +/- 600 centipawns (6 pawns)
    eval_score = (predicted_value - 0.5) * 1200

    return eval_score


use_nn_eval = True

def get_piece_color(piece_str):
    if piece_str is None: return None
    return piece_str[0]

def is_valid_square(row, col):
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

def get_opponent_color(color):
    return 'b' if color == 'w' else 'w'

def get_pawn_moves(board_pychess, row, col):
    piece_color = chess.WHITE if board_pychess.piece_at(chess.square(col, 7-row)).color == chess.WHITE else chess.BLACK
    moves = []
    square = chess.square(col, 7-row) # Convert Pygame row/col to chess square index
    for move in board_pychess.legal_moves:
        if move.from_square == square and board_pychess.piece_at(square).piece_type == chess.PAWN:
            to_row = 7 - (move.to_square // 8)
            to_col = move.to_square % 8
            moves.append((to_row, to_col))
    return moves

def get_rook_moves(board_pychess, row, col):
    moves = []
    square = chess.square(col, 7-row)
    for move in board_pychess.legal_moves:
        if move.from_square == square and board_pychess.piece_at(square).piece_type == chess.ROOK:
            to_row = 7 - (move.to_square // 8)
            to_col = move.to_square % 8
            moves.append((to_row, to_col))
    return moves

def get_knight_moves(board_pychess, row, col):
    moves = []
    square = chess.square(col, 7-row)
    for move in board_pychess.legal_moves:
        if move.from_square == square and board_pychess.piece_at(square).piece_type == chess.KNIGHT:
            to_row = 7 - (move.to_square // 8)
            to_col = move.to_square % 8
            moves.append((to_row, to_col))
    return moves

def get_bishop_moves(board_pychess, row, col):
    moves = []
    square = chess.square(col, 7-row)
    for move in board_pychess.legal_moves:
        if move.from_square == square and board_pychess.piece_at(square).piece_type == chess.BISHOP:
            to_row = 7 - (move.to_square // 8)
            to_col = move.to_square % 8
            moves.append((to_row, to_col))
    return moves

def get_queen_moves(board_pychess, row, col):
    moves = []
    square = chess.square(col, 7-row)
    for move in board_pychess.legal_moves:
        if move.from_square == square and board_pychess.piece_at(square).piece_type == chess.QUEEN:
            to_row = 7 - (move.to_square // 8)
            to_col = move.to_square % 8
            moves.append((to_row, to_col))
    return moves

def get_king_moves(board_pychess, row, col):
    moves = []
    square = chess.square(col, 7-row)
    for move in board_pychess.legal_moves:
        if move.from_square == square and board_pychess.piece_at(square).piece_type == chess.KING:
            to_row = 7 - (move.to_square // 8)
            to_col = move.to_square % 8
            moves.append((to_row, to_col))
    return moves

def get_pseudo_legal_moves_for_piece(board_pychess, row, col):
     square = chess.square(col, 7-row)
     piece = board_pychess.piece_at(square)
     if piece is None: return []

     moves = []
     for move in board_pychess.pseudo_legal_moves: # Check pseudo first
         if move.from_square == square:
             to_row = 7 - (move.to_square // 8)
             to_col = move.to_square % 8
             moves.append((to_row, to_col))
     return moves


def find_king(board_pychess, color):
    king_square = board_pychess.king(color)
    if king_square is None: return None
    return (7 - (king_square // 8), king_square % 8)


def is_square_attacked(board_pychess, target_row, target_col, attacker_color):
     square_index = chess.square(target_col, 7-target_row)
     return board_pychess.is_attacked_by(attacker_color, square_index)


def is_in_check(board_pychess, color):
     return board_pychess.is_check() and board_pychess.turn == color


def get_all_legal_moves(board_pychess):
    legal_moves_dict = {}
    for move in board_pychess.legal_moves:
        from_r, from_c = 7 - (move.from_square // 8), move.from_square % 8
        to_r, to_c = 7 - (move.to_square // 8), move.to_square % 8
        start_pos = (from_r, from_c)
        end_pos = (to_r, to_c)
        if start_pos not in legal_moves_dict:
            legal_moves_dict[start_pos] = []
        legal_moves_dict[start_pos].append(end_pos)
    return legal_moves_dict, bool(legal_moves_dict) # Return dict and whether any moves exist


def make_move(board_pychess, start_pos, end_pos):
    global move_history, game_state, promotion_pos, promotion_color
    start_row, start_col = start_pos
    end_row, end_col = end_pos

    from_square = chess.square(start_col, 7-start_row)
    to_square = chess.square(end_col, 7-end_row)

    piece = board_pychess.piece_at(from_square)
    if not piece: return False, None # Should not happen if called with legal move

    piece_color = piece.color
    piece_type = piece.piece_type
    needs_promotion = False

    # Check for promotion *before* creating the move object if player needs choice
    last_rank = 7 if piece_color == chess.WHITE else 0 # Note: python-chess ranks are 0-7
    if piece_type == chess.PAWN and end_row == last_rank:
        if (piece_color == chess.WHITE and player_color == 'w') or \
           (piece_color == chess.BLACK and player_color == 'b'):
            needs_promotion = True
            promotion_pos = (end_row, end_col)
            promotion_color = 'w' if piece_color == chess.WHITE else 'b'
            # Construct move *without* promotion for now, choice happens later
            move = chess.Move(from_square, to_square)
        else: # AI promotes to Queen
            move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
    else:
         move = chess.Move(from_square, to_square)


    if move in board_pychess.legal_moves:
        captured_piece = board_pychess.piece_at(to_square) # Get piece before pushing move
        board_pychess.push(move)
        move_history.append(move) # Store python-chess move object
        return True, captured_piece
    else:
        # This should not happen if we only call make_move with validated moves
        print(f"Error: Illegal move attempted {move.uci()}")
        # Try finding the intended move in legal_moves again (e.g., castling notation difference)
        # This part is complex, safer to rely on correct input validation
        return False, None


def evaluate_board_ai(board_pychess):
    if board_pychess.is_insufficient_material() or board_pychess.is_stalemate():
        return 0
    if board_pychess.is_checkmate():
        return -100000 if board_pychess.turn == chess.WHITE else 100000 # Score relative to whose turn it is

    score = 0
    for square in chess.SQUARES:
        piece = board_pychess.piece_at(square)
        if piece:
            piece_type_char = piece.symbol().upper()
            piece_color_char = 'w' if piece.color == chess.WHITE else 'b'
            base_value = PIECE_VALUES.get(piece_type_char, 0)
            pos_score = 0
            if piece_type_char in POSITION_SCORES:
                r = 7 - (square // 8)
                c = square % 8
                pos_r = r if piece.color == chess.WHITE else BOARD_SIZE - 1 - r
                pos_c = c
                try:
                    pos_score = POSITION_SCORES[piece_type_char][pos_r][pos_c]
                except IndexError:
                    pos_score = 0

            current_piece_score = base_value + pos_score
            score += current_piece_score if piece.color == chess.WHITE else -current_piece_score
    return score

def get_material_score(board_pychess, color):
    score = 0
    for square in chess.SQUARES:
        piece = board_pychess.piece_at(square)
        if piece and piece.color == color:
             piece_type = piece.piece_type
             if piece_type != chess.KING:
                  score += PIECE_VALUES.get(piece.symbol().upper(), 0)
    return score // 100

def score_move(board_pychess, move):
    score = 0
    # Promotion bonus
    if move.promotion is not None:
        score += PIECE_VALUES.get(chess.piece_symbol(move.promotion).upper(), 900) # Base on promoted piece value

    # Capture bonus (MVV-LVA)
    if board_pychess.is_capture(move):
        attacker = board_pychess.piece_at(move.from_square)
        victim = board_pychess.piece_at(move.to_square)
        if victim is None: # Handle en passant captures
             if board_pychess.is_en_passant(move):
                  victim_value = PIECE_VALUES['P']
             else: victim_value = 0 # Should not happen?
        else:
             victim_value = PIECE_VALUES.get(victim.symbol().upper(), 0)

        attacker_value = PIECE_VALUES.get(attacker.symbol().upper(), 0)
        score += 1000 + (victim_value * 10) - attacker_value

    return score


def minimax(board_pychess, depth, alpha, beta, maximizing_player, start_time, time_limit):
    global use_nn_eval
    if time.time() - start_time > time_limit:
         raise TimeoutError

    if board_pychess.is_game_over():
        if board_pychess.is_checkmate():
            return (-math.inf if maximizing_player else math.inf), None
        else: # Draw (stalemate, insufficient material, etc.)
            return 0, None

    if depth == 0:
        eval_func = evaluate_board_nn if use_nn_eval else evaluate_board_ai
        return eval_func(board_pychess), None

    moves = list(board_pychess.legal_moves)
    moves.sort(key=lambda m: score_move(board_pychess, m), reverse=True)

    best_move = None

    if maximizing_player:
        max_eval = -math.inf
        for move in moves:
            board_pychess.push(move)
            try:
                evaluation, _ = minimax(board_pychess, depth - 1, alpha, beta, False, start_time, time_limit)
            except TimeoutError:
                 board_pychess.pop()
                 raise TimeoutError
            board_pychess.pop()

            if evaluation > max_eval:
                max_eval = evaluation
                best_move = move
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for move in moves:
            board_pychess.push(move)
            try:
                evaluation, _ = minimax(board_pychess, depth - 1, alpha, beta, True, start_time, time_limit)
            except TimeoutError:
                 board_pychess.pop()
                 raise TimeoutError
            board_pychess.pop()

            if evaluation < min_eval:
                min_eval = evaluation
                best_move = move
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        return min_eval, best_move


def get_ai_move(board_pychess, time_limit):
    global use_nn_eval
    start_time = time.time()
    depth = 0
    best_move_overall = None
    best_score_overall = -math.inf if board_pychess.turn == chess.WHITE else math.inf
    is_maximizing = board_pychess.turn == chess.WHITE

    print(f"AI ({'W' if is_maximizing else 'B'}) thinking... (FEN: {board_pychess.fen()}) (Eval: {'NN' if use_nn_eval else 'Classic'})")

    try:
        while True:
            depth += 1
            elapsed_time = time.time() - start_time
            remaining_time = time_limit - elapsed_time

            if remaining_time <= 0.1: # Need more buffer time
                 print(f"Stopping iterative deepening at depth {depth-1}: Low time remaining.")
                 break
            if elapsed_time > time_limit * 0.8 and depth > 2: # Stop if most time used
                 print(f"Stopping iterative deepening at depth {depth-1}: >80% time used.")
                 break

            print(f"  Trying depth {depth}...")
            current_depth_start_time = time.time()

            try:
                score, move = minimax(board_pychess.copy(), depth, -math.inf, math.inf, is_maximizing, start_time, time_limit)
                if move is not None: # Check if a valid move was found for this depth
                    best_move_overall = move
                    best_score_overall = score
                    print(f"  Depth {depth} completed. Best move: {move.uci()}, Score: {score:.2f} (Time: {time.time() - current_depth_start_time:.2f}s)")
                else:
                    print(f"  No move returned at depth {depth}, likely game over found.")
                    # If no move found, the previous depth's move is the best we have
                    if best_move_overall is None:
                         print(" Error: No move found even at low depth.")
                    break # Stop deepening

            except TimeoutError:
                 print(f"  Timeout occurred during depth {depth}. Using best move from depth {depth-1}.")
                 break # Stop deepening

    except Exception as e:
        print(f"An error occurred during AI move generation: {e}")
        traceback.print_exc()
        if best_move_overall:
             print("Falling back to last completed depth move due to error.")
        else:
             print("Error and no previous move found. AI cannot move.")
             return None

    final_time = time.time() - start_time
    print(f"AI finished in {final_time:.2f}s. Chose move: {best_move_overall.uci() if best_move_overall else 'None'} (Score: {best_score_overall:.2f})")

    if best_move_overall is None:
        print("AI Error: No move selected after search. Choosing random move.")
        legal_moves_final = list(board_pychess.legal_moves)
        if legal_moves_final:
            return random.choice(legal_moves_final)
        else:
             print("AI Error: No legal moves available for random choice.")
             return None

    return best_move_overall



def board_list_to_pychess(board_list):
    board_pc = chess.Board(fen=None) # Create empty board
    board_pc.clear_board()
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece_str = board_list[r][c]
            if piece_str:
                color = chess.WHITE if piece_str[0] == 'w' else chess.BLACK
                piece_type_char = piece_str[1].upper()
                piece_type = chess.PIECE_SYMBOLS.index(piece_type_char.lower())
                square = chess.square(c, 7-r)
                board_pc.set_piece_at(square, chess.Piece(piece_type, color))

    # Set turn and other game state attributes from the global board_pychess if available.
    board_pc.turn = chess.WHITE if current_turn == 'w' else chess.BLACK

    if board_pychess is not None:
        # Sync with the main game board state.
        board_pc.castling_rights = board_pychess.castling_rights
        board_pc.ep_square = board_pychess.ep_square
        board_pc.halfmove_clock = board_pychess.halfmove_clock
        board_pc.fullmove_number = board_pychess.fullmove_number
    else:
        # Fallback to default values if global board_pychess is not initialized.
        board_pc.castling_rights = 0
        board_pc.ep_square = None
        board_pc.halfmove_clock = 0
        board_pc.fullmove_number = 1 # Standard for a new board after setup.

    return board_pc

def pychess_to_board_list(board_pychess):
     new_board_list = [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
     for r in range(BOARD_SIZE):
          for c in range(BOARD_SIZE):
               square = chess.square(c, 7-r)
               piece = board_pychess.piece_at(square)
               if piece:
                    color_char = 'w' if piece.color == chess.WHITE else 'b'
                    type_char = piece.symbol().upper()
                    new_board_list[r][c] = color_char + type_char
     return new_board_list

def to_display_coords(r, c, p_color):
    if p_color == 'b':
        display_r, display_c = BOARD_SIZE - 1 - r, c
    else:
        display_r, display_c = r, c
    return (display_c * SQUARE_SIZE, display_r * SQUARE_SIZE)

def to_internal_coords(screen_x, screen_y, p_color):
    display_c = screen_x // SQUARE_SIZE
    display_r = screen_y // SQUARE_SIZE
    if not (0 <= display_c < BOARD_SIZE and 0 <= display_r < BOARD_SIZE):
         return None
    if p_color == 'b':
        return (BOARD_SIZE - 1 - display_r, display_c)
    else:
        return (display_r, display_c)

def draw_board(surface, p_color):
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            screen_x, screen_y = to_display_coords(r, c, p_color)
            color = LIGHT_SQUARE if (r + c) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(surface, color, (screen_x, screen_y, SQUARE_SIZE, SQUARE_SIZE))

def draw_pieces(surface, board_state, p_color, animating_piece_info=None):
    anim_r_start, anim_c_start = -1, -1
    if animating_piece_info:
        anim_r_start, anim_c_start = animating_piece_info['start']

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if animating_piece_info and r == anim_r_start and c == anim_c_start:
                continue

            piece = board_state[r][c]
            if piece is not None:
                if piece in piece_images:
                    screen_x, screen_y = to_display_coords(r, c, p_color)
                    img = piece_images[piece]
                    img_rect = img.get_rect(topleft=(screen_x, screen_y))
                    surface.blit(img, img_rect)
                else:
                    pass

def highlight_square(surface, r, c, color, p_color):
    screen_x, screen_y = to_display_coords(r, c, p_color)
    highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    highlight_surf.fill(color)
    surface.blit(highlight_surf, (screen_x, screen_y))

def draw_game_state(surface, current_board_pychess, sel_piece_coords, p_color, animating_piece_info):
    # Use current_board_pychess to draw
    board_list = pychess_to_board_list(current_board_pychess) # Convert for drawing
    draw_board(surface, p_color)

    if current_board_pychess.is_check():
        king_sq = current_board_pychess.king(current_board_pychess.turn)
        if king_sq is not None:
            king_r, king_c = 7 - (king_sq // 8), king_sq % 8
            highlight_square(surface, king_r, king_c, CHECK_HIGHLIGHT_COLOR, p_color)

    if sel_piece_coords and not is_animating and game_state != 'PROMOTION':
         # Convert python-chess square index to row/col if needed, or use stored row/col
         start_r, start_c = sel_piece_coords
         highlight_square(surface, start_r, start_c, HIGHLIGHT_COLOR, p_color)
         # Highlight legal moves for selected piece
         try:
             from_square = chess.square(start_c, 7-start_r)
             for move in current_board_pychess.legal_moves:
                 if move.from_square == from_square:
                      to_r, to_c = 7 - (move.to_square // 8), move.to_square % 8
                      highlight_square(surface, to_r, to_c, POSSIBLE_MOVE_COLOR, p_color)
         except Exception as e:
              print(f"Error highlighting moves: {e}")


    draw_pieces(surface, board_list, p_color, animating_piece_info) # Use board_list for drawing

def draw_material_display_and_forfeit(surface):
    global player_color, forfeit_button_rect, status_message, board_pychess

    area_y = GAME_AREA_HEIGHT
    pygame.draw.rect(surface, CAPTURED_BG_COLOR, (0, area_y, WIDTH, CAPTURED_AREA_HEIGHT))
    margin = 10
    text_y_center = area_y + CAPTURED_AREA_HEIGHT // 2

    if "thinking..." in status_message:
        try:
            think_surf = font_status.render(status_message, True, CAPTURED_TEXT_COLOR)
            think_rect = think_surf.get_rect(center=(WIDTH // 2, text_y_center))
            surface.blit(think_surf, think_rect)
        except Exception as e:
            print(f"Error rendering thinking status: {e}")
        return

    if player_color is None or board_pychess is None: return

    player_chess_color = chess.WHITE if player_color == 'w' else chess.BLACK
    ai_chess_color = chess.BLACK if player_color == 'w' else chess.WHITE

    player_score = get_material_score(board_pychess, player_chess_color)
    ai_score = get_material_score(board_pychess, ai_chess_color)

    player_label = f"Player ({player_color.upper()}): {player_score}"
    ai_label = f"AI ({('W' if ai_chess_color == chess.WHITE else 'B')}): {ai_score}"

    try:
        player_surf = font_captured.render(player_label, True, CAPTURED_TEXT_COLOR)
        player_rect = player_surf.get_rect(midleft=(margin, text_y_center))
        surface.blit(player_surf, player_rect)

        ai_surf = font_captured.render(ai_label, True, CAPTURED_TEXT_COLOR)
        ai_rect = ai_surf.get_rect(right=(WIDTH - margin - forfeit_button_rect.width - margin), centery=text_y_center)
        surface.blit(ai_surf, ai_rect)

        pygame.draw.rect(surface, FORFEIT_BUTTON_COLOR, forfeit_button_rect, border_radius=5)
        forfeit_text = font_button.render("Forfeit", True, FORFEIT_TEXT_COLOR)
        forfeit_text_rect = forfeit_text.get_rect(center=forfeit_button_rect.center)
        surface.blit(forfeit_text, forfeit_text_rect)

    except Exception as e:
        print(f"Error rendering material/forfeit display: {e}")


def draw_promotion_choices(surface):
    global promotion_choice_rects
    overlay = pygame.Surface((WIDTH, GAME_AREA_HEIGHT), pygame.SRCALPHA)
    overlay.fill(PROMOTION_BG_COLOR)
    surface.blit(overlay, (0,0))

    text_surf = font_status.render("Choose promotion:", True, PROMOTION_TEXT_COLOR)
    text_rect = text_surf.get_rect(center=(WIDTH // 2, GAME_AREA_HEIGHT // 2 - 60))
    surface.blit(text_surf, text_rect)

    choice_size = SQUARE_SIZE
    total_width = len(promotion_choices) * choice_size + (len(promotion_choices) - 1) * 10
    start_x = (WIDTH - total_width) // 2
    start_y = GAME_AREA_HEIGHT // 2 - choice_size // 2

    promotion_choice_rects = []
    for i, piece_char in enumerate(promotion_choices):
        rect = pygame.Rect(start_x + i * (choice_size + 10), start_y, choice_size, choice_size)
        pygame.draw.rect(surface, PROMOTION_CHOICE_COLOR, rect, border_radius=5)
        pygame.draw.rect(surface, PROMOTION_BORDER_COLOR, rect, 2, border_radius=5)

        piece_surf = font_promotion.render(piece_char, True, PROMOTION_TEXT_COLOR)
        piece_rect = piece_surf.get_rect(center=rect.center)
        surface.blit(piece_surf, piece_rect)
        promotion_choice_rects.append(rect)


def draw_game_over(surface, message):
    overlay = pygame.Surface((WIDTH, GAME_AREA_HEIGHT), pygame.SRCALPHA)
    overlay.fill((50, 50, 50, 180))
    surface.blit(overlay, (0,0))
    text_surface = font_game_over.render(message, True, (200, 0, 0))
    text_rect = text_surface.get_rect(center=(WIDTH // 2, GAME_AREA_HEIGHT // 2 - 20))
    surface.blit(text_surface, text_rect)
    restart_text_surf = font_restart.render("Press 'R' to return to Menu", True, WHITE)
    restart_rect = restart_text_surf.get_rect(center=(WIDTH // 2, GAME_AREA_HEIGHT // 2 + 30))
    surface.blit(restart_text_surf, restart_rect)

def draw_menu(surface):
    surface.fill(MENU_BG_COLOR)
    title_surf = font_menu_title.render("Chess Game", True, MENU_TEXT_COLOR)
    title_rect = title_surf.get_rect(center=(WIDTH // 2, HEIGHT // 4))
    surface.blit(title_surf, title_rect)
    option_w_surf = font_menu_option.render("Play as White (Press W)", True, MENU_TEXT_COLOR)
    option_w_rect = option_w_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    surface.blit(option_w_surf, option_w_rect)
    option_b_surf = font_menu_option.render("Play as Black (Press B)", True, MENU_TEXT_COLOR)
    option_b_rect = option_b_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 60))
    surface.blit(option_b_surf, option_b_rect)

def reset_game():
    global board_pychess, selected_square, legal_moves_for_selected
    global game_over, winner, status_message
    global is_animating, pending_move, move_history
    global promotion_pos, promotion_color, game_state

    board_pychess = chess.Board() # Use python-chess board
    selected_square = None
    legal_moves_for_selected = []
    game_over = False
    winner = None
    status_message = ""
    move_history = []
    is_animating = False
    pending_move = None
    promotion_pos = None
    promotion_color = None
    game_state = 'PLAYING' # Assume reset goes directly to playing
    print("Game reset.")

def start_animation(start_r, start_c, end_r, end_c, piece_str, p_color):
    global is_animating, anim_piece_img, anim_start_pos_screen, anim_end_pos_screen
    global anim_current_pos_screen, anim_progress, pending_move_uci

    if piece_str not in piece_images:
         print(f"Error starting animation: Invalid piece string '{piece_str}'")
         return

    is_animating = True
    from_square = chess.square(start_c, 7-start_r)
    to_square = chess.square(end_c, 7-end_r)
    pending_move_uci = chess.Move(from_square, to_square).uci() # Store UCI for applying later

    anim_piece_img = piece_images[piece_str]
    anim_start_pos_screen = to_display_coords(start_r, start_c, p_color)
    anim_end_pos_screen = to_display_coords(end_r, end_c, p_color)
    anim_current_pos_screen = anim_start_pos_screen
    anim_progress = 0

board_pychess = None # Initialize python-chess board representation
selected_square = None # Use chess square index for selection
legal_moves_for_selected = [] # Store potential end squares for selected piece

running = True
while running:
    current_time = time.time()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            running = False

        current_player_chess_color = chess.WHITE if current_turn == 'w' else chess.BLACK

        if game_state == 'MENU':
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    player_color = 'w'
                    reset_game() # This sets game_state = 'PLAYING'
                elif event.key == pygame.K_b:
                    player_color = 'b'
                    reset_game()

        elif game_state == 'PROMOTION':
             if event.type == pygame.MOUSEBUTTONDOWN:
                 mouse_x, mouse_y = event.pos
                 for i, rect in enumerate(promotion_choice_rects):
                     if rect.collidepoint(mouse_x, mouse_y):
                         chosen_piece_char = promotion_choices[i]
                         promo_r, promo_c = promotion_pos # Pygame coords
                         promo_square = chess.square(promo_c, 7-promo_r)

                         # Need the original move to add promotion info
                         # Assuming pending_move_uci holds the move like 'e7e8'
                         original_move = chess.Move.from_uci(pending_move_uci)
                         promotion_piece_type = chess.PIECE_TYPES[promotion_choices.index(chosen_piece_char)+1] # Q=5, R=4, B=3, N=2

                         move_with_promotion = chess.Move(
                             original_move.from_square,
                             original_move.to_square,
                             promotion=promotion_piece_type
                         )

                         if move_with_promotion in board_pychess.legal_moves:
                             board_pychess.push(move_with_promotion)
                             move_history.append(move_with_promotion)
                             print(f"Player promoted to {chosen_piece_char} via {move_with_promotion.uci()}")
                         else:
                             # Fallback: Manually set piece if move fails (shouldn't happen often)
                             print(f"Warning: Promotion move {move_with_promotion.uci()} not legal? Manually setting piece.")
                             promo_color_chess = chess.WHITE if promotion_color == 'w' else chess.BLACK
                             board_pychess.set_piece_at(promo_square, chess.Piece(promotion_piece_type, promo_color_chess))


                         game_state = 'PLAYING'
                         current_turn = get_opponent_color(current_turn)
                         promotion_pos = None
                         promotion_color = None
                         pending_move_uci = None # Clear pending move

                         if board_pychess.is_checkmate():
                             game_over = True; winner = get_opponent_color(current_turn); status_message = f"Checkmate! {'White' if winner == 'w' else 'Black'} wins!"
                         elif board_pychess.is_stalemate() or board_pychess.is_insufficient_material() or board_pychess.is_seventyfive_moves() or board_pychess.is_fivefold_repetition():
                             game_over = True; winner = 'stalemate'; status_message = "Draw!"
                         elif board_pychess.is_check():
                              status_message = f"{'White' if current_turn == 'w' else 'Black'} is in Check!"
                         else:
                              status_message = ""
                         break

        elif game_state in ['PLAYING', 'GAME_OVER']:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game_state = 'MENU'; player_color = None; is_animating = False; board_pychess = None

            if game_state == 'PLAYING' and not game_over and (board_pychess.turn == chess.WHITE if player_color == 'w' else board_pychess.turn == chess.BLACK) and not is_animating:
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos

                    if forfeit_button_rect.collidepoint(mouse_x, mouse_y):
                        game_over = True
                        winner = get_opponent_color(player_color)
                        status_message = f"{'White' if player_color == 'w' else 'Black'} forfeited. {'White' if winner == 'w' else 'Black'} wins!"
                        print(status_message)
                        game_state = 'GAME_OVER'

                    elif mouse_y < GAME_AREA_HEIGHT:
                        coords = to_internal_coords(mouse_x, mouse_y, player_color)
                        if coords:
                            clicked_row, clicked_col = coords
                            clicked_square = chess.square(clicked_col, 7-clicked_row)

                            if selected_square is not None:
                                # Try to make the move
                                move = chess.Move(selected_square, clicked_square)
                                # Check if promotion needed
                                piece = board_pychess.piece_at(selected_square)
                                needs_promotion_choice = False
                                if piece and piece.piece_type == chess.PAWN:
                                     last_rank = 7 if piece.color == chess.WHITE else 0
                                     if chess.square_rank(clicked_square) == last_rank:
                                          needs_promotion_choice = True

                                if needs_promotion_choice:
                                     # Check if base move (without promotion) is pseudo-legal first
                                     temp_move_no_promo = chess.Move(selected_square, clicked_square)
                                     if board_pychess.is_legal(temp_move_no_promo): # Check if the move is legal before asking for choice
                                        needs_promotion = True
                                        promotion_pos = (clicked_row, clicked_col) # Store pygame coords
                                        promotion_color = player_color
                                        pending_move_uci = temp_move_no_promo.uci() # Store base move UCI
                                        game_state = 'PROMOTION'
                                        selected_square = None
                                        legal_moves_for_selected = []
                                        status_message = "Choose promotion..."

                                     else: # Move itself wasn't legal base
                                          # If click was on another own piece, select it
                                          clicked_piece_obj = board_pychess.piece_at(clicked_square)
                                          if clicked_piece_obj and clicked_piece_obj.color == current_player_chess_color:
                                               selected_square = clicked_square
                                               legal_moves_for_selected = [m.to_square for m in board_pychess.legal_moves if m.from_square == selected_square]
                                          else: # Clicked invalid square or opponent piece
                                               selected_square = None
                                               legal_moves_for_selected = []

                                elif move in board_pychess.legal_moves:
                                    piece_str = board_pychess.piece_at(selected_square).symbol()
                                    piece_str = ('w' if board_pychess.turn == chess.WHITE else 'b') + piece_str.upper()

                                    start_r, start_c = 7 - (move.from_square // 8), move.from_square % 8
                                    end_r, end_c = 7 - (move.to_square // 8), move.to_square % 8

                                    start_animation(start_r, start_c, end_r, end_c, piece_str, player_color)

                                    selected_square = None
                                    legal_moves_for_selected = []
                                    status_message = ""

                                else: # Move was not legal, check if selecting new piece
                                    clicked_piece_obj = board_pychess.piece_at(clicked_square)
                                    if clicked_piece_obj and clicked_piece_obj.color == current_player_chess_color:
                                         selected_square = clicked_square
                                         legal_moves_for_selected = [m.to_square for m in board_pychess.legal_moves if m.from_square == selected_square]
                                    else:
                                         selected_square = None
                                         legal_moves_for_selected = []

                            else: # No piece selected previously
                                clicked_piece_obj = board_pychess.piece_at(clicked_square)
                                if clicked_piece_obj and clicked_piece_obj.color == current_player_chess_color:
                                     selected_square = clicked_square
                                     legal_moves_for_selected = [m.to_square for m in board_pychess.legal_moves if m.from_square == selected_square]
                                else:
                                     selected_square = None
                                     legal_moves_for_selected = []


    if is_animating:
        anim_progress += 1
        t = anim_progress / ANIMATION_DURATION
        t = min(1.0, t)

        start_x, start_y = anim_start_pos_screen
        end_x, end_y = anim_end_pos_screen
        current_x = start_x + (end_x - start_x) * t
        current_y = start_y + (end_y - start_y) * t
        anim_current_pos_screen = (current_x, current_y)

        if anim_progress >= ANIMATION_DURATION:
            is_animating = False
            move = chess.Move.from_uci(pending_move_uci) # Use stored UCI

            if move in board_pychess.legal_moves:
                board_pychess.push(move)
                move_history.append(move)
            else:
                 print(f"Error: Animated move {pending_move_uci} became illegal?")
                 # Consider how to handle this rare case - maybe skip turn?

            pending_move_uci = None
            current_turn = get_opponent_color(current_turn)

            if board_pychess.is_checkmate():
                 game_over = True; winner = get_opponent_color(current_turn); status_message = f"Checkmate! {'White' if winner == 'w' else 'Black'} wins!"
                 game_state = 'GAME_OVER'
            elif board_pychess.is_stalemate() or board_pychess.is_insufficient_material() or board_pychess.is_seventyfive_moves() or board_pychess.is_fivefold_repetition():
                 game_over = True; winner = 'stalemate'; status_message = "Draw!"
                 game_state = 'GAME_OVER'
            elif board_pychess.is_check():
                 status_message = f"{'White' if current_turn == 'w' else 'Black'} is in Check!"
            else:
                 status_message = ""

    elif game_state == 'PLAYING' and not game_over and (board_pychess.turn == chess.BLACK if player_color == 'w' else board_pychess.turn == chess.WHITE):
        status_message = f"AI ({current_turn}) is thinking..."

        screen.fill(BLACK)
        draw_game_state(screen, board_pychess, selected_square, player_color, None)
        draw_material_display_and_forfeit(screen)
        pygame.display.flip()

        ai_move = get_ai_move(board_pychess, AI_THINK_TIME)
        status_message = ""

        if ai_move and ai_move in board_pychess.legal_moves:
             start_r, start_c = 7 - (ai_move.from_square // 8), ai_move.from_square % 8
             end_r, end_c = 7 - (ai_move.to_square // 8), ai_move.to_square % 8
             moved_piece_obj = board_pychess.piece_at(ai_move.from_square)
             piece_str = ('w' if moved_piece_obj.color == chess.WHITE else 'b') + moved_piece_obj.symbol().upper()

             start_animation(start_r, start_c, end_r, end_c, piece_str, player_color)
             pending_move_uci = ai_move.uci() # Store the move UCI

        else:
             print("AI Error: get_ai_move returned None or illegal move. Checking game state.")
             if board_pychess.is_checkmate():
                 game_over = True; winner = player_color ; status_message = f"Checkmate! {'White' if winner == 'w' else 'Black'} wins!"
                 game_state = 'GAME_OVER'
             elif board_pychess.is_stalemate() or board_pychess.is_insufficient_material() or board_pychess.is_seventyfive_moves() or board_pychess.is_fivefold_repetition():
                 game_over = True; winner = 'stalemate'; status_message = "Draw!"
                 game_state = 'GAME_OVER'
             else:
                  print("AI failed to move but game not over? Turn might be skipped incorrectly.")
                  # This path suggests an issue in AI or game state validation
                  # Maybe force forfeit? Or just log error. For now, log and potentially hangs.


    screen.fill(BLACK)
    if game_state == 'MENU':
        draw_menu(screen)
    elif game_state in ['PLAYING', 'PROMOTION', 'GAME_OVER']:
        king_check_color = None
        if board_pychess and not game_over and board_pychess.is_check():
             king_check_color = 'w' if board_pychess.turn == chess.WHITE else 'b'

        animating_piece_info = None
        if is_animating and pending_move_uci:
             # Extract start pos from pending_move_uci for draw exclusion
             try:
                 move_obj = chess.Move.from_uci(pending_move_uci)
                 start_r, start_c = 7 - (move_obj.from_square // 8), move_obj.from_square % 8
                 animating_piece_info = {'start': (start_r, start_c)}
             except: pass # Ignore if UCI is bad

        # Convert selected square index to row/col for draw_game_state if needed
        sel_coords = None
        if selected_square is not None:
             sel_coords = (7 - (selected_square // 8), selected_square % 8)

        if board_pychess: # Ensure board exists before drawing
             draw_game_state(screen, board_pychess, sel_coords, player_color, animating_piece_info)

             if is_animating and anim_piece_img:
                 anim_rect = anim_piece_img.get_rect(topleft=anim_current_pos_screen)
                 screen.blit(anim_piece_img, anim_rect)

             draw_material_display_and_forfeit(screen)

             if game_state == 'PROMOTION':
                  draw_promotion_choices(screen)

             if game_over or game_state == 'GAME_OVER':
                 final_message = ""
                 if "forfeited" in status_message:
                     final_message = status_message
                 elif winner == 'w': final_message = "Checkmate! White Wins!"
                 elif winner == 'b': final_message = "Checkmate! Black Wins!"
                 elif winner == 'stalemate': final_message = "Stalemate!"
                 draw_game_over(screen, final_message)
        else:
             # Should only happen briefly before reset_game is called if coming from menu
             pass


    pygame.display.flip()
    clock.tick(FPS)

print("Exiting game.")
pygame.quit()
sys.exit()