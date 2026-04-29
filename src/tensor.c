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
	int stride = 1;
	int idx = 0;
	for (int i = t->ndim - 1; i >= 0; --i)
	{
		idx += indices[i] * stride;
		stride *= t->shape[i];
	}
	return &t->data[idx];
}

void tensor_print(const Tensor* t)
{
	printf("Shape=(");
	int elements = 1;
	for (size_t i = 0; i < t->ndim; ++i)
	{
		elements *= t->shape[i];
		if (i != t->ndim - 1)
			printf("%d, ", t->shape[i]);
		else
			printf("%d", t->shape[i]);
	}
	printf("), data=[");
	for (size_t i = 0; i < elements; ++i)
	{
		if (i != elements - 1)
			printf("%f, ", t->data[i]);
		else
			printf("%f", t->data[i]);
	}
	printf("]\n");
}
