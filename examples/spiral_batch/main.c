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

	// Training parameters
	i32 epochs = 5000;
	i32 min_epochs = 1000;

	f32 beta1 = 0.9;
	f32 beta2 = 0.999;
	f32 eps = 1e-8f;
	f32 lr = 1e-3f;
	OptimParams p = { .b1 = beta1, .b2 = beta2, .eps = eps, .lr = lr };

	// NN Architecture
	// Two hidden layers with a size of 8
	// h = relu(x @ w1 + b1)
	// out = h @ w2 + b2
	i32 n_hide = 16;
	i32 N = 200;

	i32 shape_x     [] = { 2 * N  , 2			};
	i32 shape_target[] = { 2 * N  , 2			};

	i32 shape_w1    [] = { 2	  , n_hide 		};
	i32 shape_b1    [] = { 1	  , shape_w1[1] };

	i32 shape_w2    [] = { n_hide , n_hide 		};
	i32 shape_b2    [] = { 1	  , shape_w2[1] };

	i32 shape_w3    [] = { n_hide , 2			};
	i32 shape_b3	[] = { 1	  , shape_w3[1] };

	// Spiral example
	Tensor* x = tensor_create(arena_p, shape_x, 2, true);
	Tensor* target = tensor_zeros(arena_p, shape_target, 2);
	data_generate_spiral_ce(N, x, target);


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
	f32 tol = 1e-2f;

	// Batch trainingg
	i32 batch_size = 64;
	i32* indices = PUSH_ARRAY(arena_p, i32, 2 * N);
	for (i32 i = 0; i < 2 * N; ++i)
	{
		indices[i] = i;
	}
	i32 batches = (i32)(2 * N / batch_size);

	// Global step variable
	i32 step = 1;

	// Training loop
	for (i32 epoch = 0; epoch < epochs; ++epoch)
	{
		data_shuffle(indices, 2 * N);
		f32 batch_loss = 0.0f;
		for (i32 it = 0; it < batches; ++it)
		{
			i32 batch_start = batch_size * it;
			Tensor* x_batch = tensor_gather(arena_t, x, indices + batch_start, batch_size);
			Tensor* t_batch = tensor_gather(arena_t, target, indices + batch_start, batch_size);
			
			// FORWARD PASS (Cross Entropy loss function)
			Tensor *h1 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x_batch, w1), b1));
			Tensor *h2 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, h1, w2), b2));
			Tensor *out = graph_add(arena_t, graph_matmul(arena_t, h2, w3), b3);
			Tensor *loss = graph_softmax_ce(arena_t, out, t_batch);
			batch_loss += loss->data[0];

			// BACKPROP
			backward(arena_t, loss);

			// TRAIN
			adam_step(learnable, sizeof(learnable) / sizeof(learnable[0]), &p, step++);

			// CLEAR INTERMEDIATES
			arena_clear(arena_t);
		}
		batch_loss /= batches;
		if (epoch % (epochs / 10) == 0)
		{
			printf("epoch %d: loss = %f\n", epoch, batch_loss);
		}
		if (batch_loss < tol && epoch > min_epochs)
			break;
	}

	printf("\n==========================================\n");
	printf("=================RESULTS==================");
	printf("\n==========================================\n");

	data_generate_spiral_ce(N, x, target);

	Tensor* h1 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x, w1), b1));
	Tensor* h2 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, h1, w2), b2));
	Tensor* out = graph_add(arena_t, graph_matmul(arena_t, h2, w3), b3);
	Tensor* loss = graph_softmax_ce(arena_t, out, target);

	Tensor* weights[6] = { w1, w2, w3, b1, b2, b3 };
	Tensor* ms[6] = { aw1.m, aw2.m, aw3.m, ab1.m, ab2.m, ab3.m };
	Tensor* vs[6] = { aw1.v, aw2.v, aw3.v, ab1.v, ab2.v, ab3.v };
	data_save_tensors(weights, sizeof(weights) / sizeof(weights[0]), "../../../data/weights/spiral_batch_weights.csv");
	data_save_tensors(ms, sizeof(weights) / sizeof(weights[0]), "../../../data/weights/spiral_batch_ms.csv");
	data_save_tensors(vs, sizeof(weights) / sizeof(weights[0]), "../../../data/weights/spiral_batch_vs.csv");

	Tensor* plotting[2] = { x, out };
	data_save_tensors(plotting, sizeof(plotting) / sizeof(plotting[0]), "../../../data/spiral_batch_data.csv");

	printf("Final loss: ");
	tensor_print(loss);

	arena_destroy(arena_p);
	arena_destroy(arena_t);
}