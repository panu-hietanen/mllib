import argparse
import os

from mllib.data import preprocess_data

def parse_args():
    p = argparse.ArgumentParser(description="Train chess data preprocessing")
    p.add_argument("--data_path",    default="~/dev/mllib/train_data/random_evals.csv")
    p.add_argument("--save_path",    default="~/dev/mllib/train_data/random_evals_preprocessed", help="Path prefix for saved weights")
    return p.parse_args()

def main():
    args = parse_args()
    data_path = os.path.expanduser(args.data_path)
    save_path = os.path.expanduser(args.save_path)
    preprocess_data(data_path, save_path)

if __name__ == "__main__":
    main()