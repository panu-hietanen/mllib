#include "tensor.h"

Tensor* tensor_create(mem_arena* arena, int* shape, int ndim, bool non_zero)
{
	u64 size = 1;
	for (size_t i = 0; i < ndim; ++i)
	{
		size *= shape[i];
	}
	Tensor* t = arena_push(arena, sizeof(Tensor), true);
	t->data = arena_push(arena, size * sizeof(f32), non_zero);
	t->ndim = ndim;
	memcpy(t->shape, shape, ndim * sizeof(int));
	return t;
}

f32* tensor_at(const Tensor* t, int* indices)
{
	int idx = 0;
	for (size_t i = 0; i < t->ndim; ++i)
	{
		int stride = 1;
		for (size_t j = i + 1; j < t->ndim; ++j)
		{
			stride *= t->shape[j];
		}
		idx += indices[i] * stride;
	}
	return &t->data[idx];
}
