import numpy as np

from mllib.nn import Model, Linear, ReLU
from mllib.data import load_chunk, FEATURES

epochs = 3
min_epochs = 1
report_epochs = 1

chunk_size = 1000
tol = 0
data_path = "~/dev/mllib/train_data/random_evals.csv"

hidden_dims = 256
input_dims = FEATURES
output_dims = 1

model = Model(
    layers=[Linear(input_dims, hidden_dims), ReLU(), Linear(hidden_dims, output_dims)],
    loss="sigmoid_bce",
    lr=1e-3,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    arena_size=1024 * 1024 * 64
)

for epoch in range(epochs):
    epoch_loss = 0.0
    n = 0
    n_chunks = 0
    done = False
    while not done:
        X, y = load_chunk(data_path, skip=n, n=chunk_size)
        if len(X) == 0 or len(y) == 0:
            done = True
            break
        epoch_loss += model.forward(X, y)
        if (n_chunks % 100 == 0):
            print(f"epoch {epoch}, chunk {n_chunks}: loss = {epoch_loss / (n_chunks + 1)}")

        model.backward()
        model.step()
        n += chunk_size
        n_chunks += 1
    epoch_loss /= n_chunks
    if epoch_loss < tol and epoch > min_epochs:
        break
    if epoch % report_epochs == 0:
        print(f"epoch {epoch}: overall epoch loss = {epoch_loss:.4f}")
    print(f"####################EPOCH {epoch} COMPLETE####################")
