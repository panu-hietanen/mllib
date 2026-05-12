import os

from mllib.data import preprocess_data

def main():
    data_path = os.path.expanduser("~/dev/mllib/train_data/")
    preprocess_data(data_path + "random_evals.csv", data_path + "random_evals_preprocessed")

if __name__ == "__main__":
    main()