import argparse
import os
import numpy as np

from mllib.nn import Model, Linear, ReLU
from mllib.data import load_chunk, load_preprocessed_chunk, FEATURES

def parse_args():
    p = argparse.ArgumentParser(description="Train chess evaluation network")
    p.add_argument("--data",         default="~/dev/mllib/train_data/random_evals.csv")
    p.add_argument("--epochs",       type=int,   default=3)
    p.add_argument("--chunk-size",   type=int,   default=1000)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--hidden",       type=int,   default=256)
    p.add_argument("--preprocessed", action="store_true", help="Use preprocessed .npy files; pass path prefix to --data, not the CSV")
    p.add_argument("--no-save",      action="store_true", help="Disable saving weights after each epoch")
    p.add_argument("--save-path",    default="~/dev/mllib/data/weights/chess_weights", help="Path prefix for saved weights")
    p.add_argument("--load",         default=None,                                     help="Path prefix to resume from")
    return p.parse_args()

def main():
    args = parse_args()
    data_path = os.path.expanduser(args.data)
    save_path = os.path.expanduser(args.save_path)
    
    model = Model(
        layers=[Linear(FEATURES, args.hidden), ReLU(), Linear(args.hidden, 1)],
        loss="sigmoid_bce",
        lr=args.lr,
        arena_size=1024 * 1024 * 64,
    )

    if args.load:
        print(f"Resuming from {args.load}")
        model.load(args.load)

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n = 0
        n_chunks = 0
        while True:
            if args.preprocessed:
                path_indices = data_path + "_X.npy"
                path_labels = data_path + "_y.npy"
                X, y = load_preprocessed_chunk(path_indices, path_labels, skip=n, n=args.chunk_size)
            else:
                X, y = load_chunk(data_path, skip=n, n=args.chunk_size)
            if len(X) == 0:
                break
            epoch_loss += model.forward(X, y)
            model.backward()
            model.step()

            if n_chunks % 100 == 0:
                print(f"epoch {epoch}, chunk {n_chunks}: loss = {epoch_loss / (n_chunks + 1):.4f}")
            n += args.chunk_size
            n_chunks += 1

        epoch_loss /= n_chunks
        print(f"epoch {epoch} complete: average loss = {epoch_loss:.4f}")
        if not args.no_save:
            model.save(save_path)
            print(f"saved to {save_path}")

if __name__ == "__main__":
    main()
