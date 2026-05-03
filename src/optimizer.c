#include "optimizer.h"

void step(Tensor** weights, i32 n, f32 lr)
{
	for (i32 t = 0; t < n; ++t)
	{
		Tensor* curr = weights[t];
		i32 elements = tensor_number_elements(curr);
		for (i32 i = 0; i < elements; ++i)
		{
			curr->data[i] -= lr * curr->grad->data[i];
		}
		tensor_fill(curr->grad, 0.0f);
	}
}
