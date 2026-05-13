---
title: Devlog
---

# mllib Devlog

My experience building a machine learning library in C from scratch, with the long-term goal of training a chess evaluation network.

---

## Background

The project started with a clear goal: use my theoretical machine learning background to implement a deep learning library in C. I wanted it to be capable of forward passes, backpropagation, and training loops, similar to PyTorch. I also was interested in how you can mix the two languages, so I wanted to try to make some Python bindings to use the C code under the hood.

---

## 1 Architecture Planning

Some of the decisions I made early on in the project really defined how I was going to be using memory etc. so had to be well motivated.

**Memory management strategy**

The first major decision was how to handle memory. Really there are two approaches here:

- **Reference counting**: every tensor tracks how many things point to it and frees itself when the count hits zero
- **Arena allocation**: one large slab of memory, allocated upfront, handed out by bumping a pointer. I had seen this before in a video made by [Magicalbat](https://www.youtube.com/@Magicalbat), who really inspired this project for me.

I chose Arena allocaton for its simplicity. Reference counting seems easier to have memory leaks in, and with training going over hours then you really want to avoid this. An arena means one allocator owns everything and you free it all at once. This maps naturally to ML training, where you allocate a computation graph and then can get rid of a lot of the data after doing a training loop.

**Project structure**

I considered a single file approach, where we avoid using headers for simplicity. The chosen approach was one `.c`/`.h` file per module as usual, as I am used to this when programming in C++ and it made it easier to plan out the project in my head as I went.

```
src/
  arena.c / arena.h
  tensor.c / tensor.h
  node.h
  graph.c / graph.h
  optimizer.c / optimizer.h
  data.c / data.h
```

CMake was already set up from the start, and I found that making a single `config.h` file with global imports and defines helped prevent any missing functions from the STL.

---

## 2 Arena Allocator

### Initial implementation

The arena struct tracks capacity and current position within the slab:

```c
typedef struct {
    u64 capacity;
    u64 pos;
} mem_arena;
```

Rather than storing a separate base pointer, the struct itself lives at the start of the allocated block, and `ARENA_BASE = sizeof(mem_arena)` is used as the initial offset. This means one `malloc` call gives both the struct and the data region.

### Alignment

`arena_push` aligns each allocation to `ARENA_ALIGN = sizeof(void*)`. This alignment makes the behaviour more consistent. The position is rounded up before each allocation.

### Virtual memory experiment

I thought about replacing `malloc` with Windows `VirtualAlloc`, reserving a large address space and committing pages on demand. I didn't do this in the end because it was too tied to the operating system, and we'll see later why this is important.

### Pop operations

`arena_pop` and `arena_pop_to` were added for flexibility, though in reality `arena_clear` (reset to base) is the main operation used between training steps.

---

## 3 Tensor

### Struct design

```c
typedef struct Tensor {
    f32* data;
    i32 shape[MAX_DIMS];
    i32 ndim;
    struct Tensor* grad;
    Node* node;
    bool visited;
} Tensor;
```

Key decisions:
- `shape` is a fixed array `i32[MAX_DIMS]`, not a pointer. This was simpler for me and in reality I will be using 2D tensors for most of the training anyway.
- `grad` and `node` fields were added later when autograd was implemented
- `visited` was added for topological sort during backprop

### Indexing

Flat indexing: for a tensor of shape `[d0, d1, ..., dn]`, element `[i0, i1, ..., in]` is at flat index `sum(i_k * stride_k)` where `stride_k = product of all dimensions after k`. This means you don't have to have an array-of-arrays and is speedier.

The `tensor_at` function computes this in a single backwards pass, as you can accumulate the stride by multiplying the current dimension at each step:

```c
for (int i = t->ndim - 1; i >= 0; --i) {
    idx += indices[i] * stride;
    stride *= t->shape[i];
}
```

### Weight initialisation

Three initialisation functions were implemented:
- `tensor_zeros` / `tensor_ones`: constant fill
- `tensor_rand`: uniform random in `[-1, 1]`
- `tensor_xavier`: a common random initialisation

I found that Xavier initialisation was important for an early test case I did, training a network on XOR. Early runs with uniform random initialisation would sometimes just start predicting `0.5` for all inputs and get stuck in a local minimum. Xavier init with zeros for biases fixed this.

### Broadcasting in tensor_add

When adding tensors of different shapes (e.g. `[4, 4] + [1, 4]` for bias addition), broadcasting is needed. My implementation converts the flat output index to `[row, col]`, then use `shape == 1 ? 0 : row/col` to map back to input indices:

```c
i32 aidx = (a->shape[0] == 1 ? 0 : r) * a->shape[1] + (a->shape[1] == 1 ? 0 : c);
i32 bidx = (b->shape[0] == 1 ? 0 : r) * b->shape[1] + (b->shape[1] == 1 ? 0 : c);
```

This handles all broadcast cases (row broadcast, column broadcast, scalar broadcast) with no special cases. The same trick is applied in `add_backward` where the broadcast dimensions can accumulate gradients to the same index.

---

## 4 Forward Ops

### tensor_matmul

For now the standard triple-loop matrix multiply. Asserts `ndim == 2`, as I mentioned before the the library intentionally stays 2D to keep it simple. Extending this in the future would be possible but I'm not sure how crucial it is at the moment.

### tensor_relu

Simple elementwise max with zero:
```c
new->data[i] = (a->data[i] > 0) ? a->data[i] : 0;
```

### tensor_mse

Mean squared error over all elements:
```c
loss = sum((pred - target)²) / n
```
Output is a scalar `[1, 1]` tensor.

### tensor_softmax

Row-wise softmax for batched inputs of shape `[N, C]`. We can also use a trick where we take away the row maximum before taking the exponent. This makes it more stable and doesn't affect the result (you can prove it!).

---

## 5 Autograd

### Computation graph

Each tensor optionally holds a `Node*` pointer which points to the Tensor's parents. The `Node` struct has:

```c
typedef struct {
    Tensor* inputs[2];
    void (*backward)(mem_arena*, const Tensor*);
    void* aux;
} Node;
```

Interesting challenge: How do we store things like whether the data array was positive when doing ReLU? I chose to do this using the `void* aux` field, where I stored op-specific data needed for backward passes. E.g. for ReLU this is a boolean mask tensor indicating which elements were positive.

### Graph wrapper functions

A key decision I made: the ops in `tensor.c` are only dealing with matrices, and so the nodes are not written at this point. We need some way to have graph knowledge. Therefore I created wrapper functions in `graph.c` that both perform the op and attach a node:

```c
Tensor* graph_matmul(mem_arena* arena, Tensor* a, Tensor* b) {
    Tensor* c = tensor_matmul(arena, a, b);
    Node* node = PUSH_STRUCT(arena, Node);
    node->inputs[0] = a;
    node->inputs[1] = b;
    node->backward = matmul_backward;
    c->node = node;
    return c;
}
```

### Backward functions

**add_backward**: gradient of a sum is 1 for both inputs so accumulate `t->grad` into both input grads. For broadcast dimensions, multiple output positions map to the same input position, so `+=` naturally sums them.

**matmul_backward**: for `C = A @ B`:
- `dA = dC @ B^T`
- `dB = A^T @ dC`

I wrote both out fully rather than calling `tensor_matmul` because of the transpose. It avoids allocating intermediate tensors and accumulates directly with `+=`.

**relu_backward**: the mask stored in `node->aux` records which inputs were positive. Gradient passes through where mask is 1, zero otherwise.

**mse_backward**: Here the loss only has a single input value, as the targets do not have a gradient associated.
### Topological sort and backward()

`backward()` uses depth-first search to build an ordered list of tensors, then iterates in reverse:

1. `visit()` recursively follows `node->inputs`, appending each tensor after its dependencies
2. `backward()` initialises the loss tensor's grad to ones
3. Iterates the list in reverse, calling `node->backward` for each non-leaf tensor
4. Resets `visited` flags for the next call

Grad tensors are lazily initialised to zeros on the first backward pass through each parameter.

---

## 6 Optimizer

### SGD

```c
w->data[i] -= lr * w->grad->data[i];
```

Simple but sensitive to learning rate and initialisation.

### Adam

Tracks first moment (momentum) and second moment (variance) per parameter:

```c
m = b1 * m + (1 - b1) * grad
v = b2 * v + (1 - b2) * grad²
w -= lr * (m / (1 - b1^t)) / (sqrt(v / (1 - b2^t)) + eps)
```

The `AdamWeight` struct groups a weight tensor with its moment tensors:
```c
typedef struct {
    Tensor* w;
    Tensor* m;
    Tensor* v;
} AdamWeight;
```

Moment tensors live on the permanent arena and across training steps. Adam made training significantly more consistent, as SGD would occasionally diverge or converge slowly depending on initialisation.

---

## 7 Two-Arena Training Pattern

I realised that we need to preserve some data across iterations. Therefore decided to use two arenas instead of a single one, meaning one can be cleared for stuff that doesn't matter (like the current forward pass), whilst the weights can be preserved. 

```c
mem_arena* arena_p = arena_create(MiB(1));  // permanent: weights, biases, moments
mem_arena* arena_t = arena_create(MiB(8));  // temporary: activations, nodes, grads
```

At the end of each iteration:
```c
arena_clear(arena_t);
```

This wipes all intermediate tensors in a single pointer reset, whereas weight tensors on `arena_p` are unaffected.

---

## 8 Training Results

### XOR

A 2-layer MLP (`[2, 8, 1]` with ReLU) trained on XOR converged reliably to near-perfect accuracy:
```
Shape=(4, 1), data=[-0.000000, 1.000000, 1.000000, -0.000000]
```

As mentioned before, it would occasionally get stuck at loss=0.25 (predicting 0.5 for all inputs). Fixed by:
- Xavier initialisation for weights
- Zero initialisation for biases
- Increasing hidden size to 8
- Adam optimizer

### Spiral classification

A 3-layer MLP (`[2, 16, 16, 1]`) trained on a two-class spiral dataset. Data generated procedurally with `data_generate_spiral` using a small amount noise.

```
iteration 0:      loss = 0.713
iteration 10000:  loss = 0.004
iteration 90000:  loss = 0.003
```

Results exported to CSV and plotted in Python with matplotlib, which helped to visually show that the training was working.

---

## 9 Data I/O

### Weight saving

`data_save_tensors` writes tensors to a text file, one per line:
```
ndim,shape[0],shape[1],...,data[0],data[1],...
```
Uses `%.9g` format to preserve full `f32` precision.

### Weight loading

`data_load_weights` reads the same format, allocating fresh tensors on the arena with shapes read from the file. The caller doesn't need to know tensor shapes in advance because we saved it in the file beforehand.
### Plot export

Training results (input coordinates + predictions) exported to CSV and read by a Python script for matplotlib visualisation. This was easier than using any libraries in C and shows the advantage of using both languages together. Python has some really useful libraries that allow you to focus on doing the things that are actually useful and interesting.

---

## 10 Project Structure and Cross-Platform

### Examples restructure

`main.c` was moved to an examples folder, with the library compiled as a static library:

```
src/          — library code (compiled once as mllib_core)
examples/
  spiral/     — MSE spiral example
  spiral_ce/  — cross-entropy spiral
```

CMake macro for adding examples:
```cmake
macro(add_example name)
    add_executable(${name} examples/${name}/main.c)
    target_link_libraries(${name} PRIVATE mllib_core)
endmacro()
```

### Linux/SSH support

At this point I had also had an idea to begin building a remote server myself. This was going to be on my home network with Linux, and so I had to port some of the building over.

The project was extended to build on a remote Ubuntu server (GCC 15.2.0) via VSCode SSH. Changes required:
- Added `linux-debug` and `linux-release` CMake presets using Ninja
- Added `target_link_libraries(mllib_core PUBLIC m)` for `libm` on Linux
- Fixed ASAN flags, where MSVC uses `/fsanitize=address`, GCC uses `-fsanitize=address` with an additional link flag

I may make something in the future explaining how I went about this project and some of the interesting things it has let me do.

---

## 11 Completing the Op Set

### Sigmoid and BCE

`tensor_sigmoid` computes elementwise `1 / (1 + exp(-x))`.

Interesting design challenge: There is an inherent advantage to using sigmoid and BCE, as well as softmax and CE seen later. These however require the graph to know what the upstream node is, to check whether the optimisation can be applied. I decided to therefore fuse these layers into a single operation, which allows me to perform these optimisations easily. The aux field of the node came in handy here, allowing me to store the intermediate softmax result.

`sigmoid_bce_backward` can then calculate the fused gradient `(sigmoid(x) - y) / N` directly. This is faster and more numerically stable than computing sigmoid and BCE loss separately.

### Fused softmax+CE

`graph_softmax_ce` computes softmax then cross-entropy in one node. The softmax probabilities are stored in `aux` for use in the backward pass. The fused backward gradient is `(softmax(x) - y) / N`.

### Mini-batch support

`tensor_gather` selects rows by index, used to implement mini-batch training where a random subset of rows is selected each iteration using Fisher-Yates shuffle.

`adam_step_flat` is a flat-array variant of Adam that takes three `Tensor**` arrays (weights, m moments, v moments). I made this to make calling from Python ctypes easier.

### tensor_set_data / tensor_get_data

Two new functions for copying data in/out of tensors via raw float pointers. These are the bridge for Python to C data transfer, where we can just store the pointer to the Tensor in Python and then operate on this using C functions called using ctypes.

---

## 12 Python ctypes Bindings

### Architecture

The Python bindings were added in `python/mllib/` with three layers:

1. **`_bindings.py`**: the `lib.*` ctypes declarations with `argtypes`/`restype`, plus typed wrapper functions that make calling them easier. The shared library is compiled separately as `libmllib.so` (CMake `SHARED` target).

2. **`tensor.py`**: `Arena` and `Tensor` Python wrapper classes. Arena holds a `_ptr` to the C arena. `Tensor.from_numpy` converts a numpy array to a C tensor via `tensor_set_data`.

3. **`nn.py`**: PyTorch-style layer classes:
   - `Linear`: initialises xavier weights and zero biases, zero Adam moments, all on `arena_p`
   - `ReLU`, `Sigmoid`: stateless wrappers around graph ops
   - `Model`: owns both arenas, collects parameters from all Linear layers, builds ctypes pointer arrays for `adam_step_flat`

### Model design

The two-arena pattern is hidden inside `Model`. This makes the API a lot simpler as the user never has to deal with the arenas.

```python
model = Model(
    layers=[Linear(2, 16), ReLU(), Linear(16, 2)],
    loss="softmax_ce",
    lr=1e-3
)
loss = model.forward(x, target)
model.backward()
model.step()
```

`adam_step_flat` is called with three ctypes pointer arrays built from `self.ws`, `self.ms`, `self.vs`:
```python
self._ws_arr = (ctypes.c_void_p * n)(*[t._ptr for t in self.ws])
```

### Spiral training from Python

`python/training.py` trains the spiral classifier from Python. Python handles the data generation and defines the model, then it calls the C library to operate on the graph. Loss converges to ~0.02 over 5000 epochs, showing Python to C to Python pipeline works correctly.

---

## 13 Chess Data Pipeline

### Dataset

Three datasets downloaded from Kaggle chess-evaluations:
- `chessData.csv`: 12.9M positions, format: `FEN,Evaluation`
- `random_evals.csv`: 1M positions, same format. I used this for development iteration to confirm training worked
- `tactic_evals.csv`: 2.6M positions, has extra `Move` column and mostly mate scores. I skipped it for now as this is for an engine that actually chooses the best move

Evaluations are centipawn strings: `+56`, `-9`. Mate scores `#+N` are filtered out as this is already handled by the game.

### Feature encoding with 781 features

I used the python chess library to help with parsing FEN strings. This allowed me to process the datasets easily, and then read them into numpy arrays. I had to make sure to align the piece definitions with what I was going to use in my own chess game, as I made up the ordering myself.
- **Features 0–767**: piece positions (6 piece types x 2 colours x 64 squares)
  - `piece_id = PIECE_TO_TYPE[piece_type] + colour_offset` (white offset=0, black offset=6)
  - `PIECE_TO_TYPE`: PAWN=0, ROOK=1, KNIGHT=2, BISHOP=3, QUEEN=4, KING=5
  - `feature_idx = piece_id * 64 + square`
- **Feature 768**: side to move (1=white, 0=black)
- **Features 769–772**: castling rights (wK, wQ, bK, bQ)
- **Features 773–780**: en passant file (one-hot over 8 files)

Having this encoding allows you to uniquely represent the board state, which was important for the NN to know how to evaluate the position. E.g. it can make a big difference to the evaluation if you are able to castle or not.

---

## 14 Chess Training

### Training script

`python/training_chess.py` is a CLI script with argparse:

```bash
python3 python/training_chess.py --epochs 3 --data ~/dev/mllib/train_data/random_evals.csv
python3 python/training_chess.py --epochs 3 --load ~/dev/mllib/data/weights/chess_weights
python3 python/training_chess.py --preprocessed --data ~/dev/mllib/train_data/random_evals_preprocessed
```

Flags: `--data`, `--epochs`, `--chunk-size`, `--lr`, `--hidden`, `--preprocessed`, `--no-save`, `--save-path`, `--load`.

### Network architecture

Input 781 -> Hidden 256 (ReLU) -> Output 1 (sigmoid+BCE). Trained with Adam at lr=1e-3, chunk size 1000.

### Convergence

Training on `random_evals.csv` (944k positions after filtering):
- Epoch 0 start: loss = 0.694 (near random)
- Epoch 0 end: loss = 0.661
- Epoch 1 end: loss ~0.615

---

## 15 Performance Profiling and Matmul Optimisation

### Profiling setup

`perf stat` and `perf record` were used to profile the training loop. I did this as I initially did some preprocessing of the data as I thought this was a bottleneck, but it turned out to barely change the training time.

Manual timing with `time.perf_counter()` around each training step gave:

```
load:     ~0.10s   (6%)
forward:  ~0.25s  (14%)
backward: ~1.45s  (79%)
step:     ~0.006s  (1%)
```

`perf record` confirmed the breakdown:
```
78.71%  matmul_backward
13.34%  tensor_matmul (forward)
 2.76%  Python interpreter
```

`matmul_backward` calls `tensor_matmul` twice (grad w.r.t. inputs and grad w.r.t. weights), which accounts for the ~6:1 ratio vs the forward pass. This means we need to concentrate our efforts on the O(n^3) loop that performs matrix multiplication.

### Loop reorder optimisation

The original `tensor_matmul` used `i, j, k` loop order. In the inner loop, `b->data[k * width + j]` increments `k` by a full row width, which is a cache miss on every iteration for large matrices.

Reordered to `i, k, j`:
```c
for (i32 i = 0; i < a->shape[0]; ++i)
for (i32 k = 0; k < a->shape[1]; ++k)
{
    f32 a_ik = a->data[i * a->shape[1] + k];
    for (i32 j = 0; j < b->shape[1]; ++j)
        new->data[i * new->shape[1] + j] += a_ik * b->data[k * b->shape[1] + j];
}
```

`a_ik` is taken out as a scalar, `b` and `C` are accessed sequentially in the inner loop, all cache-friendly.

**Result**: 100 iterations went from 3m10s to 2m45s (~13% improvement). Modest because the compiler had already partially optimised the original.

### Backwards pass

We can do the same for the backwards pass, which has its own matmul loops. Maybe this shows that I shouldn't have made them separately however it isnt too much code to change. Note that the transpose means that one of the loops need to be in a different order. 

### Compiler

We can tell the compiler that we are accessing memory that doesn't overlap (isn't aliased) using the restrict keyword. This allows it to perform vectorised operations.

We can also use the `-march=native` flag to ensure that the binary is optimised for the CPU that we are compiling it on. As I am training it on the machine I am compiling it on and not using the GPU, it made sense to use this

### Outcome
All of these optimisations reduced the training time further to 35s, which was a huge improvement. Overall we have reduced the training from 3m10s to 35s just by changing some compiler flags and loop ordering. I suspect that the compiler flags provided the biggest increase in performance.

### Next optimisation: tiling

The next step is blocking/tiling, processing submatrices that fit in L1 cache. A tile size of 32 or 64 is typical. This is kept as a future improvement alongside the option of linking OpenBLAS (`cblas_sgemm`) for a larger speedup. I'm hesitant to link OpenBLAS as I would like to do these things myself, but maybe I could take some inspiration.

---

## Lessons Learned

- **Arena allocation is a simple but effective tool**: It is easy to setup and allows makes handling memory leaks much easier.
- **Keeping ops pure and graph wrappers separate**: Having a structure that encapsulates different parts of the code helps to make the extensions clearer in your head.
- **Profiling before optimising**: The assumed bottleneck of the data processing was wrong, which I only found out after I had implemented a solution. Profiling with `perf` showed that the matrix multiplication was the biggest culprit and addressing it gave huge performance gains.
- **Loop order matters for cache**: Reordering from `i,j,k` to `i,k,j` in matmul gives sequential memory access in the inner loop