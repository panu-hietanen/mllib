import os
import ctypes
import numpy as np
import sys
from math import pi, sin, cos

from mllib.nn import Model, Linear, ReLU
from mllib.tensor import Tensor


n_datapoints = 400
epochs = 5000
min_epochs = 500
tol = 1e-3

x = np.zeros((n_datapoints, 2), dtype=np.float32)
target = np.zeros((n_datapoints, 2), dtype=np.float32)

offset = [0, pi]
for i in range(n_datapoints // 2):
    t = 4 * pi * i / (n_datapoints // 2)
    for j in range(2):
        noise = (np.random.rand() * 2 - 1) * 0.2
        row = i + j * (n_datapoints // 2)
        x[row] = [(t * cos(t + offset[j]) + noise), (t * sin(t + offset[j]) + noise)] 
        target[row, j] = 1.0

hidden_dims = 16
model = Model(
    layers=[Linear(2, hidden_dims), ReLU(), Linear(hidden_dims, hidden_dims), ReLU(), Linear(hidden_dims, 2)],
    loss="softmax_ce",
    lr=1e-3,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    arena_size=1024 * 1024 * 64
)

batch_size = 64
batches = n_datapoints // batch_size

from mllib.data import load_chunk

X, y = load_chunk("train_data/chessData.csv", skip=0, n=100)
print(X.shape, y.shape)
print(y[:5])

indices = np.arange(0, n_datapoints)
for epoch in range(epochs):
    epoch_loss = 0.0
    np.random.shuffle(indices)
    for i in range(batches):
        batch_start = i * batch_size
        batch_end = (i + 1) * batch_size
        batch_indices = indices[batch_start:batch_end]
        x_batch = x[batch_indices]
        t_batch = target[batch_indices]
        epoch_loss += model.forward(x_batch, t_batch)

        model.backward()
        model.step()
    epoch_loss /= batches
    if epoch > min_epochs and epoch_loss < tol:
        break
    if epoch % 500 == 0:
        print(f"epoch {epoch}: loss = {epoch_loss:.4f}")

save_path = os.path.expanduser("~/dev/mllib/data/weights/test_weights")
model.save(save_path)

final_loss = model.forward(x, target)
print(f"Final loss = {final_loss:.4f}")

model2 = Model(
    layers=[Linear(2, hidden_dims), ReLU(), Linear(hidden_dims, hidden_dims), ReLU(), Linear(hidden_dims, 2)],
    loss="softmax_ce",
    lr=1e-3,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    arena_size=1024 * 1024 * 64
)

model2.load(save_path)

final_loss = model2.forward(x, target)
print(f"Final loss (after reloading weights) = {final_loss:.4f}")