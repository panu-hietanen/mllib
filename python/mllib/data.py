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

def _build_mirror_perm() -> NDArray:
    perm = np.arange(FEATURES, dtype=np.int32)
    for f in range(PIECE_FEATURES):
        piece_id = f // 64
        square = f % 64
        perm[f] = ((piece_id + 6) % 12) * 64 + (square % 8) + (7 - square // 8) * 8
    # Castling: swap white/black
    perm[769] = 771
    perm[770] = 772
    perm[771] = 769
    perm[772] = 770
    return perm

def _build_piece_mirror() -> NDArray:
    mirror = np.zeros(PIECE_FEATURES, dtype=np.int32)
    for f in range(PIECE_FEATURES):
        piece_id = f // 64
        square = f % 64
        mirror[f] = ((piece_id + 6) % 12) * 64 + (square % 8) + (7 - square // 8) * 8
    return mirror

_MIRROR_PERM    = _build_mirror_perm()
_PIECE_MIRROR   = _build_piece_mirror()
_CASTLING_REMAP = np.array([771, 772, 769, 770], dtype=np.int32)

def _soft_label(evals_cp: NDArray) -> NDArray:
    return (1.0 / (1.0 + np.exp(-evals_cp.astype(np.float64) / 200.0))).astype(np.float32)

def _mirror_dense(X: NDArray, y: NDArray) -> tuple[NDArray, NDArray]:
    X_m = X[:, _MIRROR_PERM].copy()
    X_m[:, 768] = 1.0 - X_m[:, 768]
    return X_m, (1.0 - y).astype(np.float32)

def _mirror_sparse(indices: NDArray) -> NDArray:
    result = indices.astype(np.int32).copy()

    piece_mask = (indices >= 0) & (indices < 768)
    castling_mask = (indices >= 769) & (indices <= 772)
    side_mask = (indices == 768)

    result[piece_mask] = _PIECE_MIRROR[indices[piece_mask].astype(np.int32)]
    result[castling_mask] = _CASTLING_REMAP[indices[castling_mask].astype(np.int32) - 769]
    result[side_mask] = -1

    had_side = np.any(side_mask, axis=1)
    for i in np.where(~had_side)[0]:
        empty = np.where(result[i] == -1)[0]
        if len(empty) > 0:
            result[i, empty[0]] = 768

    return result.astype(np.int16)

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
            
def load_chunk(path: str, skip: int, n: int, mirror: bool) -> tuple[NDArray, NDArray]:
    df = pd.read_csv(path, skiprows=range(1, skip+1), nrows=n, header=0, dtype={"Evaluation": str})
    df_no_mate = df[~df["Evaluation"].str.startswith('#')]

    if len(df_no_mate) == 0:
        return np.zeros((0, FEATURES), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    X = np.array([fen_to_features(fen) for fen in df_no_mate["FEN"]])

    evals = df_no_mate["Evaluation"].astype(float).values.reshape(-1, 1)
    y = _soft_label(evals)

    if mirror:
        X_m, y_m = _mirror_dense(X, y)
        return np.vstack([X, X_m]), np.vstack([y, y_m])
    else:
        return X, y

def preprocess_data(path_read: str, path_save: str) -> None:
    df = pd.read_csv(path_read, header=0, dtype={"Evaluation": str})
    df_no_mate = df[~df["Evaluation"].str.startswith('#')]

    X = np.array([fen_to_compact(fen) for fen in df_no_mate["FEN"]])
    np.save(path_save + "_X", X)

    evals = df_no_mate["Evaluation"].astype(float).values.reshape(-1, 1).astype(np.float32)
    np.save(path_save + "_y", evals)

def _expand_sparse(indices: NDArray) -> NDArray:
    batch_size = len(indices)
    X = np.zeros((batch_size, FEATURES), dtype=np.float32)
    rows = np.repeat(np.arange(batch_size), indices.shape[1])
    cols = indices.flatten().astype(np.int32)
    mask = cols != -1
    X[rows[mask], cols[mask]] = 1.0
    return X

def load_preprocessed_chunk(indices_path: str, labels_path: str, skip: int, n: int, mirror: bool) -> tuple[NDArray, NDArray]:
    indices = np.load(indices_path, mmap_mode='r')[skip:skip+n]
    raw_evals = np.load(labels_path,  mmap_mode='r')[skip:skip+n]
    batch_size = len(indices)
    if batch_size == 0:
        return np.zeros((0, FEATURES), dtype=np.float32), np.zeros((0, 1), dtype=np.float32)

    y   = _soft_label(raw_evals)

    X   = _expand_sparse(indices)

    if mirror:
        indices_m = _mirror_sparse(indices)
        y_m = (1.0 - y).astype(np.float32)
        X_m = _expand_sparse(indices_m)
        return np.vstack([X, X_m]), np.vstack([y, y_m])
    else:
        return X, y