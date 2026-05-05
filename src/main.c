#include "config.h"
#include <time.h>

#include "arena.h"
#include "tensor.h"
#include "graph.h"
#include "optimizer.h"
#include "data.h"

int main()
{
	// Initialise arenas and random seed
	mem_arena* arena_p = arena_create(MiB(10));
	mem_arena* arena_t = arena_create(MiB(10));
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
	i32 n_hide  = 16;
	i32 N = 200;

	i32 shape_x     [] = { 2 * N  , 2			};

	i32 shape_w1    [] = { 2	  , n_hide 		};
	i32 shape_b1    [] = { 1	  , shape_w1[1] };

	i32 shape_w2    [] = { n_hide , n_hide 		};
	i32 shape_b2    [] = { 1	  , shape_w2[1] };

	i32 shape_w3    [] = { n_hide , 1			};
	i32 shape_b3	[] = { 1	  , shape_w3[1] };

	i32 shape_target[] = { 2 * N  , 1			};

	// XOR example
	Tensor* x = tensor_create(arena_p, shape_x, 2, true);
	Tensor* target = tensor_create(arena_p, shape_target, 2, true);
	data_generate_spiral(N, x, target);


	// Initialise weights and biases
	Tensor* w1 = tensor_xavier(arena_p, shape_w1, 2);
	Tensor* w2 = tensor_xavier(arena_p, shape_w2, 2);
	Tensor* w3 = tensor_xavier(arena_p, shape_w3, 2);
	Tensor* b1 = tensor_zeros (arena_p, shape_b1, 2);
	Tensor* b2 = tensor_zeros (arena_p, shape_b2, 2);
	Tensor* b3 = tensor_zeros (arena_p, shape_b3, 2);

	// Create extra tensors for adam gradient steps
	AdamWeight aw1 = { w1, tensor_zeros(arena_p, shape_w1, 2), tensor_zeros(arena_p, shape_w1, 2) };
	AdamWeight aw2 = { w2, tensor_zeros(arena_p, shape_w2, 2), tensor_zeros(arena_p, shape_w2, 2) };
	AdamWeight aw3 = { w3, tensor_zeros(arena_p, shape_w3, 2), tensor_zeros(arena_p, shape_w3, 2) };
	AdamWeight ab1 = { b1, tensor_zeros(arena_p, shape_b1, 2), tensor_zeros(arena_p, shape_b1, 2) };
	AdamWeight ab2 = { b2, tensor_zeros(arena_p, shape_b2, 2), tensor_zeros(arena_p, shape_b2, 2) };
	AdamWeight ab3 = { b3, tensor_zeros(arena_p, shape_b3, 2), tensor_zeros(arena_p, shape_b3, 2) };

	AdamWeight* learnable[6] = { &aw1, &aw2, &aw3, &ab1, &ab2, &ab3 };
	f32 tol = 1e-5f;

	// Training loop
	for (i32 it = 0; it < iters; ++it)
	{
		// FORWARD PASS (MSE loss function)
		Tensor* h1 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x , w1), b1));
		Tensor* h2 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, h1, w2), b2));
		Tensor* out = graph_add(arena_t, graph_matmul(arena_t, h2, w3), b3);
		Tensor* loss = graph_mse(arena_t, out, target);

		if (it % (iters / 10) == 0) 
		{
			printf("iteration %d: ", it);
			tensor_print(loss);
		}

		// BACKPROP
		backward(arena_t, loss);

		// TRAIN
		adam_step(learnable, sizeof(learnable) / sizeof(learnable[0]), &p, it + 1);

		if (loss->data[0] < tol) break;
		// CLEAR INTERMEDIATES
		arena_clear(arena_t);
	}

	printf("\n==========================================\n");
	printf("=================RESULTS==================");
	printf("\n==========================================\n");
	Tensor* h1 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x, w1), b1));
	Tensor* h2 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, h1, w2), b2));
	Tensor* out = graph_add(arena_t, graph_matmul(arena_t, h2, w3), b3);
	Tensor* loss = graph_mse(arena_t, out, target);

	printf("Final loss: ");
	tensor_print(loss);

	arena_destroy(arena_p);
	arena_destroy(arena_t);
}