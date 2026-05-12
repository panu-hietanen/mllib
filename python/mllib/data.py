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
FEATURES = 768

def fen_to_features(fen: str) -> NDArray:
    board = chess.Board(fen)
    features = np.zeros(FEATURES, dtype=np.float32)
    for idx, square in enumerate(chess.SQUARES):
        piece = board.piece_at(square)
        if piece:
            colour_offset = 0 if piece.color == chess.WHITE else 6
            piece_idx = PIECE_TO_TYPE[piece.piece_type] + colour_offset
            feature_idx = piece_idx * 64 + idx
            features[feature_idx] = 1.0
    return features
            
def load_chunk(path: str, skip: int, n: int) -> tuple[NDArray, NDArray]:
    df = pd.read_csv(path, skiprows=range(1, skip+1), nrows=n, header=0, dtype={"Evaluation": str})
    df_no_mate = df[~df["Evaluation"].str.startswith('#')]

    X = np.array([fen_to_features(fen) for fen in df_no_mate["FEN"]])
    
    evals = df_no_mate["Evaluation"].astype(float)
    y = (evals > 0).astype(np.float32).values.reshape(-1, 1)
    
    return X, y