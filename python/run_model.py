from mllib.nn import Model, Linear, ReLU
from mllib.data import fen_to_features, FEATURES
import numpy as np
import os

model = Model(
    layers=[Linear(FEATURES, 256), ReLU(), Linear(256, 1)],
    loss="sigmoid_bce", lr=1e-3, arena_size=1024*1024*64,
)
model.load(os.path.expanduser("~/dev/mllib/data/weights/chess_weights_soft"))

def eval_fen(fen):
    x = fen_to_features(fen).reshape(1, -1).astype(np.float32)
    return model.predict(x).item(0)

# Starting position — should be ~0.5
print(eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 01"))

# Same position, black to move — should be ~0.5 too (it's equal regardless of turn)
print(eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 01"))

# White up a queen (black queen removed)
print(eval_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 01"))

# Black up a queen (white queen removed)
print(eval_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 01"))
