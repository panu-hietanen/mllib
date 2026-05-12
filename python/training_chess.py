import argparse
import os
import numpy as np

from mllib.nn import Model, Linear, ReLU
from mllib.data import load_chunk, FEATURES

def parse_args():
    p = argparse.ArgumentParser(description="Train chess evaluation network")
    p.add_argument("--data",       default="~/dev/mllib/train_data/random_evals.csv")
    p.add_argument("--epochs",     type=int,   default=3)
    p.add_argument("--chunk-size", type=int,   default=1000)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden",     type=int,   default=256)
    p.add_argument("--save",       type=bool,  default=True,           help="Whether to save weights or not")
    p.add_argument("--save_path",  default="~/dev/data/chess_weights", help="Path prefix for saved weights")
    p.add_argument("--load",       default=None,                       help="Path prefix to resume from")
    return p.parse_args()

def main():
    args = parse_args()
    data_path = os.path.expanduser(args.data)
    
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
            X, y = load_chunk(data_path, skip=n, n=args.chunk_size)
            if len(X) == 0:
                break
            epoch_loss += model.forward(X, y)
            model.backward()
            model.step()
            
            if n_chunks % 100 == 0:
                print(f"epoch {epoch}, chunk {n_chunks}: loss = {epoch_loss / n_chunks:.4f}")
            n += args.chunk_size
            n_chunks += 1

        epoch_loss /= n_chunks
        print(f"epoch {epoch} complete: average loss = {epoch_loss:.4f}")
        if args.save:
            model.save(args.save_path)
            print(f"saved to {args.save_path}")

if __name__ == "__main__":
    main()
