#include "config.h"

#include "arena.h"
#include "tensor.h"
#include "graph.h"

int main()
{
	mem_arena* arena = arena_create(MiB(1));

	i32 shape_x[]   = { 1, 2 };
	i32 shape_w1[]  = { 2, 4 };
	i32 shape_w2[]  = { 4, 1 };
	i32 shape_target[] = { 1, 1 };

	Tensor* x = tensor_create(arena, shape_x, 2, true);
	x->data[0] = 5;
	x->data[1] = 3;

	Tensor* target = tensor_create(arena, shape_target, 2, true);

	Tensor* w1 = tensor_ones(arena, shape_w1, 2);
	Tensor* w2 = tensor_ones(arena, shape_w2, 2);

	Tensor* h = graph_matmul(arena, x, w1);
	Tensor* out = graph_matmul(arena, h, w2);
	Tensor* loss = graph_mse(arena, out, target);

	tensor_print(x);
	tensor_print(h);
	tensor_print(out);

	backward(arena, out);

	tensor_print(w1->grad);
	tensor_print(w2->grad);

	arena_destroy(arena);
}