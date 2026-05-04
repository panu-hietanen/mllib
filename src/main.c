#include "config.h"
#include <time.h>

#include "arena.h"
#include "tensor.h"
#include "graph.h"
#include "optimizer.h"

int main()
{
	mem_arena* arena_p = arena_create(MiB(1));
	mem_arena* arena_t = arena_create(MiB(1));
	srand(time(NULL));

	i32 iters = 1000000;
	f32 lr = 1e-2f;

	i32 shape_x[]      = { 4, 2 };
	i32 shape_w1[]     = { 2, 4 };
	i32 shape_w2[]     = { 4, 1 };
	i32 shape_b1[]     = { 1, 4 };
	i32 shape_b2[]     = { 1, 1 };
	i32 shape_target[] = { 4, 1 };

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

	Tensor* w1 = tensor_rand(arena_p, shape_w1, 2);
	Tensor* w2 = tensor_rand(arena_p, shape_w2, 2);
	Tensor* b1 = tensor_xavier(arena_p, shape_b1, 2);
	Tensor* b2 = tensor_xavier(arena_p, shape_b2, 2);
	Tensor* learnable[4] = {w1, w2, b1, b2 };

	for (i32 it = 0; it < iters; ++it)
	{
		Tensor* h = graph_relu(arena_t, graph_add(arena_t, graph_matmul(arena_t, x, w1), b1));
		Tensor* out = graph_add(arena_t, graph_matmul(arena_t, h, w2), b2);
		Tensor* loss = graph_mse(arena_t, out, target);

		if (it % (iters / 10) == 0) tensor_print(loss);

		backward(arena_t, loss);
		step(learnable, 4, lr);

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
	Tensor* h = graph_relu(arena_t, graph_matmul(arena_t, x, w1));
	Tensor* out = graph_matmul(arena_t, h, w2);

	tensor_print(out);

	arena_destroy(arena_p);
	arena_destroy(arena_t);
}