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

	// NN Architecture
	// Two hidden layers with a size of 8
	// h = relu(x @ w1 + b1)
	// out = h @ w2 + b2
	i32 N = 200;

	i32 shape_x     [] = { 2 * N  , 2			};
	i32 shape_target[] = { 2 * N  , 2			};

	// Spiral example
	Tensor* x = tensor_create(arena_p, shape_x, 2, true);
	Tensor* target = tensor_zeros(arena_p, shape_target, 2);
	data_generate_spiral_ce(N, x, target);

	printf("\n==========================================\n");
	printf("=================RESULTS==================");
	printf("\n==========================================\n");

	Tensor* weights[6] = {0};
	Tensor* ms[6] = {0};
	Tensor* vs[6] = {0};
	data_load_weights(arena_p, weights, 6, "../../../data/weights/spiral_batch_weights.csv");
	data_save_tensors(weights, sizeof(weights) / sizeof(weights[0]), "../../../data/weights/spiral_load_weights_immediate.csv");
	data_load_weights(arena_p, ms, 6, "../../../data/weights/spiral_batch_ms.csv");
	data_load_weights(arena_p, vs, 6, "../../../data/weights/spiral_batch_vs.csv");
	for (i32 i = 0; i < 6; ++i)
	{
		if (!weights[i] || !ms[i] || !vs[i])
		{
			printf("Failed to load weights — run spiral_batch first.\n");
			return 1;
		}
	}
	Tensor* w1 = weights[0];
	Tensor* w2 = weights[1];
	Tensor* w3 = weights[2];
	Tensor* b1 = weights[3];
	Tensor* b2 = weights[4];
	Tensor* b3 = weights[5];

	AdamWeight aw1 = { weights[0], ms[0], vs[0] };
	AdamWeight aw2 = { weights[1], ms[1], vs[1] };
	AdamWeight aw3 = { weights[2], ms[2], vs[2] };
	AdamWeight ab1 = { weights[3], ms[3], vs[3] };
	AdamWeight ab2 = { weights[4], ms[4], vs[4] };
	AdamWeight ab3 = { weights[5], ms[5], vs[5] };

	AdamWeight* learnable[6] = { &aw1, &aw2, &aw3, &ab1, &ab2, &ab3 };

	f32 beta1 = 0.9;
	f32 beta2 = 0.999;
	f32 eps = 1e-8f;
	f32 lr = 1e-3f;
	OptimParams p = { .b1 = beta1, .b2 = beta2, .eps = eps, .lr = lr };

	Tensor* h1 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x, w1), b1));
	Tensor* h2 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, h1, w2), b2));
	Tensor* out = graph_add(arena_t, graph_matmul(arena_t, h2, w3), b3);
	Tensor* loss = graph_softmax_ce(arena_t, out, target);

	backward(arena_t, loss);
	adam_step(learnable, sizeof(learnable) / sizeof(learnable[0]), &p, 1);
	

	Tensor* plotting[2] = { x, out };
	data_save_tensors(plotting, sizeof(plotting) / sizeof(plotting[0]), "../../../data/spiral_load_data.csv");
	data_save_tensors(weights, sizeof(weights) / sizeof(weights[0]), "../../../data/weights/spiral_load_weights.csv");

	printf("Final loss: ");
	tensor_print(loss);

	arena_destroy(arena_p);
	arena_destroy(arena_t);
}