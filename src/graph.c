#include "graph.h"

void add_backward(const Tensor* t)
{
	i32 elements = tensor_number_elements(t);
	for (i32 i = 0; i < elements; ++i)
	{
		t->node->inputs[0]->grad->data[i] += t->grad->data[i];
		t->node->inputs[1]->grad->data[i] += t->grad->data[i];
	}
}
