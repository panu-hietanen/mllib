#include "graph.h"

Tensor* graph_add(mem_arena* arena, Tensor* a, Tensor* b)
{
	Tensor* c = tensor_add(arena, a, b);
	Node* node = node_create(arena, a, b, add_backward, NULL);
	c->node = node;
	return c;
}

Tensor* graph_matmul(mem_arena* arena, Tensor* a, Tensor* b)
{
	Tensor* c = tensor_matmul(arena, a, b);
	Node* node = node_create(arena, a, b, matmul_backward, NULL);
	c->node = node;
	return c;
}

Tensor* graph_relu(mem_arena* arena, Tensor* a)
{
	Tensor* c = tensor_relu(arena, a);

	Tensor* gt_zero = tensor_create(arena, a->shape, a->ndim, true);
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		gt_zero->data[i] = (a->data[i] > 0) ? true : false;
	}

	Node* node = node_create(arena, a, NULL, relu_backward, gt_zero);
	c->node = node;
	return c;
}

Tensor* graph_mse(mem_arena* arena, Tensor* a, Tensor* b)
{
	Tensor* c = tensor_mse(arena, a, b);
	Node* node = node_create(arena, a, b, mse_backward, NULL);
	c->node = node;
	return c;
}

Tensor* graph_softmax(mem_arena* arena, Tensor* a)
{
	Tensor* c = tensor_softmax(arena, a);
	Node* node = node_create(arena, a, NULL, softmax_backward, NULL);
	c->node = node;
	return c;
}

Tensor* graph_ce(mem_arena* arena, Tensor* a, Tensor* b)
{
	Tensor* c = tensor_ce(arena, a, b);
	Node* node = node_create(arena, a, b, ce_backward, NULL);
	c->node = node;
	return c;
}

Tensor* graph_softmax_ce(mem_arena* arena, Tensor* a, Tensor* b)
{
	Tensor* softmax = tensor_softmax(arena, a);
	Tensor* c = tensor_ce(arena, softmax, b);
	Node* node = node_create(arena, a, b, softmax_ce_backward, softmax);
	c->node = node;
	return c;
}

void add_backward(mem_arena* arena, const Tensor* t)
{
	Tensor* a = t->node->inputs[0];
	Tensor* b = t->node->inputs[1];
	if (a->grad == NULL)
		a->grad = tensor_zeros(arena, a->shape, a->ndim);
	if (b->grad == NULL)
		b->grad = tensor_zeros(arena, b->shape, b->ndim);

	i32 elements = tensor_number_elements(t);
	i32 t_rows = t->shape[0];
	i32 t_cols = t->shape[1];
	for (i32 i = 0; i < elements; ++i)
	{
		i32 r = i / t_cols;
		i32 c = i % t_cols;

		i32 aidx = (a->shape[0] == 1 ? 0 : r) * a->shape[1] + (a->shape[1] == 1 ? 0 : c);
		i32 bidx = (b->shape[0] == 1 ? 0 : r) * b->shape[1] + (b->shape[1] == 1 ? 0 : c);

		a->grad->data[aidx] += t->grad->data[i];
		b->grad->data[bidx] += t->grad->data[i];
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

void relu_backward(mem_arena* arena, const Tensor* t)
{
	assert(t->grad->ndim == t->node->inputs[0]->ndim);

	Tensor* a = t->node->inputs[0];
	if (a->grad == NULL)
		a->grad = tensor_zeros(arena, a->shape, a->ndim);

	Tensor* gt_zero = (Tensor*)t->node->aux;
	i32 elements = tensor_number_elements(t);
	for (i32 i = 0; i < elements; ++i)
	{
		a->grad->data[i] += (gt_zero->data[i]) ? t->grad->data[i] : 0.0;
	}
}

void mse_backward(mem_arena* arena, const Tensor* t)
{
	assert(t->shape[0] == 1 && t->node->inputs[0]->shape[1] == 1);

	Tensor* a = t->node->inputs[0];
	Tensor* b = t->node->inputs[1];
	if (a->grad == NULL)
		a->grad = tensor_zeros(arena, a->shape, a->ndim);
	
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		a->grad->data[i] += t->grad->data[0] * 2.0f * (a->data[i] - b->data[i]) / elements;
	}
}

void softmax_backward(mem_arena* arena, const Tensor* t)
{
	assert(t->ndim == 2 && t->ndim == t->node->inputs[0]->ndim);
	for (i32 i = 0; i < t->ndim; ++i)
		assert(t->shape[i] == t->node->inputs[0]->shape[i]);

	Tensor* a = t->node->inputs[0];
	if (a->grad == NULL)
		a->grad = tensor_zeros(arena, a->shape, a->ndim);

	i32 a_rows = a->shape[0];
	i32 a_cols = a->shape[1];

	f32 dot_r[a_rows];
	for (i32 r = 0; r < a_rows; ++r)
	{
		dot_r[r] = 0.0f;
		for (i32 c = 0; c < a_cols; ++c)
		{
			i32 idx = r * a_cols + c;
			dot_r[r] += t->data[idx] * t->grad->data[idx];
		}
	}

	for (i32 r = 0; r < a_rows; ++r)
	{
		for (i32 c = 0; c < a_cols; ++c)
		{
			i32 idx = r * a_cols + c;
			a->grad->data[idx] += t->data[idx] * (t->grad->data[idx] - dot_r[r]);
		}
	}
}

void ce_backward(mem_arena* arena, const Tensor* t)
{
	Tensor* a = t->node->inputs[0];
	Tensor* b = t->node->inputs[1];
	if (a->grad == NULL)
		a->grad = tensor_zeros(arena, a->shape, a->ndim);

	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		a->grad->data[i] += t->grad->data[0] * (-b->data[i] / (a->data[i] * a->shape[0]));
	}
	
}

void softmax_ce_backward(mem_arena* arena, const Tensor* t)
{
	Tensor* a = t->node->inputs[0];
	Tensor* b = t->node->inputs[1];
	if (a->grad == NULL)
		a->grad = tensor_zeros(arena, a->shape, a->ndim);
	
	Tensor* softmax = (Tensor*)t->node->aux;
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		a->grad->data[i] += t->grad->data[0] * (softmax->data[i] - b->data[i]) / a->shape[0];
	}
}

i32 visit(Tensor** visited_list, i32 n, Tensor* t)
{
	if (t->visited) return n;
	t->visited = true;
	if (t->node != NULL)
	{
		for (i32 i = 0; i < 2; ++i)
		{
			Tensor* parent = t->node->inputs[i];
			if (parent == NULL) continue;
			n = visit(visited_list, n, parent);
		}
	}
	visited_list[n++] = t;
	return n;
}

void backward(mem_arena* arena, Tensor* t)
{
	t->grad = tensor_ones(arena, t->shape, t->ndim);
	Tensor** visited_list = PUSH_ARRAY(arena, Tensor*, MAX_NODES);
	i32 n = visit(visited_list, 0, t);
	for (i32 i = n - 1; i >= 0; --i)
	{
		Tensor* curr = visited_list[i];
		if (curr->node == NULL) continue;
		curr->node->backward(arena, curr);
	}
	for (i32 i = 0; i < n; ++i)
	{
		visited_list[i]->visited = false;
	}
}
