#include "config.h"

#include "arena.h"
#include "tensor.h"

int main()
{
	mem_arena* arena = arena_create(MiB(1));

	int shape[MAX_DIMS] = { 2, 2, 2, 0, 0, 0, 0, 0 };
	Tensor* t = tensor_create(arena, shape, 3, false);

	tensor_print(t);

	tensor_fill(t, 32);

	tensor_print(t);

	arena_destroy(arena);
	return 0;
}