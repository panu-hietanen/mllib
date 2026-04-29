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

Tensor* tensor_add(mem_arena* arena, const Tensor* a, const Tensor* b)
{
	assert(a->ndim == b->ndim);
	for (i32 i = 0; i < a->ndim; ++i)
		assert(a->shape[i] == b->shape[i]);

	Tensor* new = tensor_create(arena, a->shape, a->ndim, true);
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		new->data[i] = a->data[i] + b->data[i];
	}
	return new;
}

Tensor* tensor_mul(mem_arena* arena, const Tensor* a, f32 c)
{
	Tensor* new = tensor_create(arena, a->shape, a->ndim, true);
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		new->data[i] = a->data[i] * c;
	}
	return new;
}

Tensor* tensor_matmul(mem_arena* arena, const Tensor* a, const Tensor* b)
{
	assert(a->ndim == 2 && a->ndim == b->ndim);
	assert(a->shape[1] == b->shape[0]);

	i32 shape[MAX_DIMS] = { a->shape[0], b->shape[1] };
	Tensor* new = tensor_create(arena, shape, a->ndim, true);
	for (i32 i = 0; i < a->shape[0]; ++i)
	{
		for (i32 j = 0; j < b->shape[1]; ++j)
		{
			f32 val = 0.0;
			for (i32 k = 0; k < a->shape[1]; ++k)
			{
				i32 aidx = i * a->shape[1] + k;
				i32 bidx = k * b->shape[1] + j;
				val += a->data[aidx] * b->data[bidx];
			}
			i32 idx = i * new->shape[1] + j;
			new->data[idx] = val;
		}
	}
	return new;
}
