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

	Tensor* weights[6];
	data_load_weights(arena_p, weights, 6, "../../../data/spiral_batch_weights.csv");
	Tensor* w1 = weights[0];
	Tensor* w2 = weights[1];
	Tensor* w3 = weights[2];
	Tensor* b1 = weights[3];
	Tensor* b2 = weights[4];
	Tensor* b3 = weights[5];

	Tensor* h1 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x, w1), b1));
	Tensor* h2 = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, h1, w2), b2));
	Tensor* out = graph_add(arena_t, graph_matmul(arena_t, h2, w3), b3);
	Tensor* loss = graph_softmax_ce(arena_t, out, target);

	Tensor* plotting[2] = { x, out };
	data_save_tensors(plotting, sizeof(plotting) / sizeof(plotting[0]), "../../../data/spiral_load_weights.csv");

	printf("Final loss: ");
	tensor_print(loss);

	arena_destroy(arena_p);
	arena_destroy(arena_t);
}