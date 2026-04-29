#include "tensor.h"

Tensor* tensor_create(mem_arena* arena, const i32* shape, i32 ndim, bool non_zero)
{
	u64 size = 1;
	for (i32 i = 0; i < ndim; ++i)
	{
		size *= shape[i];
	}
	Tensor* t = arena_push(arena, sizeof(Tensor), true);
	t->data = arena_push(arena, size * sizeof(f32), non_zero);
	t->ndim = ndim;
	memcpy(t->shape, shape, ndim * sizeof(i32));
	return t;
}

f32* tensor_at(const Tensor* t, i32* indices)
{
	i32 stride = 1;
	i32 idx = 0;
	for (i32 i = t->ndim - 1; i >= 0; --i)
	{
		idx += indices[i] * stride;
		stride *= t->shape[i];
	}
	return &t->data[idx];
}

void tensor_print(const Tensor* t)
{
	printf("Shape=(");
	i32 elements = 1;
	for (i32 i = 0; i < t->ndim; ++i)
	{
		elements *= t->shape[i];
		if (i != t->ndim - 1)
			printf("%d, ", t->shape[i]);
		else
			printf("%d", t->shape[i]);
	}
	printf("), data=[");
	for (i32 i = 0; i < elements; ++i)
	{
		if (i != elements - 1)
			printf("%f, ", t->data[i]);
		else
			printf("%f", t->data[i]);
	}
	printf("]\n");
}

i32 tensor_number_elements(const Tensor* t)
{
	i32 elements = 1;
	for (i32 i = 0; i < t->ndim; ++i)
	{
		elements *= t->shape[i];
	}
	return elements;
}

void tensor_fill(Tensor* t, f32 val)
{
	i32 elements = tensor_number_elements(t);
	for (i32 i = 0; i < elements; ++i) t->data[i] = val;
}

Tensor* tensor_copy(mem_arena* arena, const Tensor* t)
{
	Tensor* new = tensor_create(arena, t->shape, t->ndim, true);
	i32 elements = tensor_number_elements(t);
	memcpy(new->data, t->data, elements * sizeof(f32));
	return new;
}

Tensor* tensor_zeros(mem_arena* arena, i32* shape, i32 ndim)
{
	Tensor* new = tensor_create(arena, shape, ndim, true);
	tensor_fill(new, 0.0);
	return new;
}

Tensor* tensor_ones(mem_arena* arena, i32* shape, i32 ndim)
{
	Tensor* new = tensor_create(arena, shape, ndim, true);
	tensor_fill(new, 1.0);
	return new;
}
