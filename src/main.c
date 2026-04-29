#include "config.h"

#include "arena.h"
#include "tensor.h"

int main()
{
	mem_arena* arena = arena_create(MiB(1));

	i32 shape_a[] = { 2, 3 };
	i32 shape_b[] = { 3, 4 };

	Tensor* a = tensor_ones(arena, shape_a, 2);
	Tensor* b = tensor_ones(arena, shape_b, 2);

	Tensor* c = tensor_matmul(arena, a, b);

	tensor_print(a);
	tensor_print(b);
	tensor_print(c);

	arena_destroy(arena);
}