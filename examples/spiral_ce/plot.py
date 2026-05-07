import numpy as np
import matplotlib.pyplot as plt

def load_tensor(line):
    values = line.strip().split(',')
    idx = 0
    ndim = int(values[idx]); idx += 1
    shape = []
    for _ in range(ndim):
        shape.append(int(values[idx])); idx += 1
    data = [float(v) for v in values[idx:]]
    return np.array(data).reshape(shape)

def load_file(filename):
    tensors = []
    with open(filename, 'r') as f:
        for line in f:
            if line.strip():
                tensors.append(load_tensor(line))
    return tensors

def main():
    # Load results
    tensors = load_file('data/spiral_ce_data.csv')
    x    = tensors[0]  # shape [2N, 2]
    pred = tensors[1]  # shape [2N, 2]

    # Plot
    pred_class = np.argmax(pred, axis=1)
    colors = ['blue' if c == 0 else 'red' for c in pred_class]

    plt.figure(figsize=(8, 8))
    plt.scatter(x[:, 0], x[:, 1], c=colors, s=10, alpha=0.7)
    plt.title('Spiral Classification')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(handles=[
        plt.scatter([], [], c='blue', label='Class 0'),
        plt.scatter([], [], c='red',  label='Class 1')
    ])
    plt.tight_layout()
    plt.savefig('plots/spiral_ce_plot.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()