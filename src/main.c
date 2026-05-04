#include "config.h"
#include <time.h>

#include "arena.h"
#include "tensor.h"
#include "graph.h"
#include "optimizer.h"

int main()
{
	// Initialise arenas and random seed
	mem_arena* arena_p = arena_create(MiB(1));
	mem_arena* arena_t = arena_create(MiB(1));
	srand(time(NULL));

	// Optimiser parameters
	i32 iters = 100000;
	f32 beta1 = 0.9;
	f32 beta2 = 0.999;
	f32 eps = 1e-8f;
	f32 lr = 1e-3f;
	OptimParams p = { .b1 = beta1, .b2 = beta2, .eps = eps, .lr = lr };

	// NN Architecture
	// Two hidden layers with a size of 8
	// h = relu(x @ w1 + b1)
	// out = h @ w2 + b2
	i32 h1_size = 8;

	i32 shape_x     [] = { 4	  , 2			};
	i32 shape_w1    [] = { 2	  , h1_size		};
	i32 shape_b1    [] = { 1	  , shape_w1[1] };
	i32 shape_w2    [] = { h1_size, 1			};
	i32 shape_b2    [] = { 1	  , shape_w2[1] };
	i32 shape_target[] = { 4	  , 1			};

	// XOR example
	Tensor* x = tensor_create(arena_p, shape_x, 2, true);
	x->data[0] = 0;
	x->data[1] = 0;
	x->data[2] = 0;
	x->data[3] = 1;
	x->data[4] = 1;
	x->data[5] = 0;
	x->data[6] = 1;
	x->data[7] = 1;

	Tensor* target = tensor_create(arena_p, shape_target, 2, true);
	target->data[0] = 0;
	target->data[1] = 1;
	target->data[2] = 1;
	target->data[3] = 0;

	// Initialise weights and biases
	Tensor* w1 = tensor_xavier(arena_p, shape_w1, 2);
	Tensor* w2 = tensor_xavier(arena_p, shape_w2, 2);
	Tensor* b1 = tensor_zeros (arena_p, shape_b1, 2);
	Tensor* b2 = tensor_zeros (arena_p, shape_b2, 2);

	// Create extra tensors for adam gradient steps
	AdamWeight aw1 = { w1, tensor_zeros(arena_p, shape_w1, 2), tensor_zeros(arena_p, shape_w1, 2) };
	AdamWeight aw2 = { w2, tensor_zeros(arena_p, shape_w2, 2), tensor_zeros(arena_p, shape_w2, 2) };
	AdamWeight ab1 = { b1, tensor_zeros(arena_p, shape_b1, 2), tensor_zeros(arena_p, shape_b1, 2) };
	AdamWeight ab2 = { b2, tensor_zeros(arena_p, shape_b2, 2), tensor_zeros(arena_p, shape_b2, 2) };

	AdamWeight* learnable[4] = { &aw1, &aw2, &ab1, &ab2 };

	// Training loop
	for (i32 it = 0; it < iters; ++it)
	{
		// FORWARD PASS (MSE loss function)
		Tensor* h = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x, w1), b1));
		Tensor* out = graph_add(arena_t, graph_matmul(arena_t, h, w2), b2);
		Tensor* loss = graph_mse(arena_t, out, target);

		if (it % (iters / 10) == 0) tensor_print(loss);

		// BACKPROP
		backward(arena_t, loss);

		// TRAIN
		adam_step(learnable, 4, &p, it + 1);

		// CLEAR INTERMEDIATES
		arena_clear(arena_t);
	}

	printf("\n==========================================\n");
	printf("=================WEIGHTS==================");
	printf("\n==========================================\n");

	tensor_print(w1);
	tensor_print(w2);
	tensor_print(b1);
	tensor_print(b2);

	printf("\n==========================================\n");
	printf("=================RESULTS==================");
	printf("\n==========================================\n");
	Tensor* h = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x, w1), b1));
	Tensor* out = graph_add(arena_t, graph_matmul(arena_t, h, w2), b2);

	tensor_print(out);

	arena_destroy(arena_p);
	arena_destroy(arena_t);
}