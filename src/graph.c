#include "graph.h"

void add_backward(mem_arena* arena, const Tensor* t)
{
	Tensor* a = t->node->inputs[0];
	Tensor* b = t->node->inputs[1];
	if (a->grad == NULL)
		a->grad = tensor_zeros(arena, a->shape, a->ndim);
	if (b->grad == NULL)
		b->grad = tensor_zeros(arena, b->shape, b->ndim);

	i32 elements = tensor_number_elements(t);
	for (i32 i = 0; i < elements; ++i)
	{
		a->grad->data[i] += t->grad->data[i];
		b->grad->data[i] += t->grad->data[i];
	}
}

void matmul_backward(mem_arena* arena, const Tensor* t)
{
	assert(t->grad->shape[1] == t->node->inputs[1]->shape[1]);
	assert(t->grad->shape[0] == t->node->inputs[0]->shape[0]);
	assert(t->grad->ndim == 2 && t->node->inputs[0]->ndim == 2 && t->node->inputs[1]->ndim == 2);

	Tensor* a = t->node->inputs[0];
	Tensor* b = t->node->inputs[1];
	if (a->grad == NULL)
		a->grad = tensor_zeros(arena, a->shape, a->ndim);
	if (b->grad == NULL)
		b->grad = tensor_zeros(arena, b->shape, b->ndim);

	for (i32 i = 0; i < t->grad->shape[0]; ++i)
	{
		for (i32 j = 0; j < b->shape[0]; ++j)
		{
			f32 val = 0.0;
			i32 cidx = i * a->grad->shape[1] + j;
			for (i32 k = 0; k < b->shape[1]; ++k)
			{
				i32 aidx = i * t->shape[1] + k;
				i32 bidx = j * b->shape[1] + k;
				val += t->grad->data[aidx] * b->data[bidx];
			}
			a->grad->data[cidx] += val;
		}
	}
	for (i32 i = 0; i < a->shape[1]; ++i)
	{
		for (i32 j = 0; j < t->grad->shape[1]; ++j)
		{
			f32 val = 0.0;
			i32 cidx = i * b->grad->shape[1] + j;
			for (i32 k = 0; k < t->grad->shape[0]; ++k)
			{
				i32 aidx = k * a->shape[1] + i;
				i32 bidx = k * t->shape[1] + j;
				val += a->data[aidx] * t->grad->data[bidx];
			}
			b->grad->data[cidx] += val;
		}
	}
}
