import numpy as np
from numpy.typing import NDArray
import pandas as pd

import chess

PIECE_TO_TYPE = {
    chess.PAWN:   0,
    chess.ROOK:   1,
    chess.KNIGHT: 2,
    chess.BISHOP: 3,
    chess.QUEEN:  4,
    chess.KING:   5,

}
PIECE_FEATURES = 768
STATE_FEATURES = 13
FEATURES = PIECE_FEATURES + STATE_FEATURES

MAX_INDICES = 40

def fen_to_features(fen: str) -> NDArray:
    board = chess.Board(fen)
    features = np.zeros(FEATURES, dtype=np.float32)

    for idx, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece:
            colour_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = PIECE_TO_TYPE[piece.piece_type] + colour_offset
            features[piece_idx * 64 + idx] = 1.0

    base = PIECE_FEATURES
    features[base]     = 1.0 if board.turn == chess.WHITE else 0.0
    features[base + 1] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    features[base + 2] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    features[base + 3] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    features[base + 4] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
    if board.ep_square is not None:
        features[base + 5 + chess.square_file(board.ep_square)] = 1.0
    return features

def fen_to_compact(fen: str) -> NDArray:
    board = chess.Board(fen)
    features = []

    for idx, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece:
            colour_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = PIECE_TO_TYPE[piece.piece_type] + colour_offset
            feature_idx = piece_idx * 64 + idx
            features.append(feature_idx)

    base = PIECE_FEATURES
    if board.turn == chess.WHITE:
        features.append(base) 
    if board.has_kingside_castling_rights(chess.WHITE):
        features.append(base + 1)
    if board.has_queenside_castling_rights(chess.WHITE):
        features.append(base + 2)
    if board.has_kingside_castling_rights(chess.BLACK):
        features.append(base + 3)
    if board.has_queenside_castling_rights(chess.BLACK):
        features.append(base + 4)
    if board.ep_square is not None:
        features.append(base + 5 + chess.square_file(board.ep_square))
    padded = np.full(MAX_INDICES, -1, dtype=np.int16)
    padded[:len(features)] = features
    return padded
            
def load_chunk(path: str, skip: int, n: int) -> tuple[NDArray, NDArray]:
    df = pd.read_csv(path, skiprows=range(1, skip+1), nrows=n, header=0, dtype={"Evaluation": str})
    df_no_mate = df[~df["Evaluation"].str.startswith('#')]

    X = np.array([fen_to_features(fen) for fen in df_no_mate["FEN"]])
    
    evals = df_no_mate["Evaluation"].astype(float)
    y = (evals > 0).astype(np.float32).values.reshape(-1, 1)
    
    return X, y

def preprocess_data(path_read: str, path_save: str) -> None:
    df = pd.read_csv(path_read, header=0, dtype={"Evaluation": str})
    df_no_mate = df[~df["Evaluation"].str.startswith('#')]

    X = np.array([fen_to_compact(fen) for fen in df_no_mate["FEN"]])
    np.save(path_save + "_X", X)
    
    evals = df_no_mate["Evaluation"].astype(float)
    y = (evals > 0).astype(np.float32).values.reshape(-1, 1)
    np.save(path_save + "_y", y)