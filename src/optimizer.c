#include "optimizer.h"

void sgd_step(Tensor** weights, i32 n, OptimParams* p)
{
	for (i32 t = 0; t < n; ++t)
	{
		Tensor* curr = weights[t];
		i32 elements = tensor_number_elements(curr);
		for (i32 i = 0; i < elements; ++i)
		{
			curr->data[i] -= p->lr * curr->grad->data[i];
		}
		tensor_fill(curr->grad, 0.0f);
	}
}

void adam_step(AdamWeight** weights, i32 n, const OptimParams* p, i32 t)
{
	for (i32 weight = 0; weight < n; ++weight)
	{
		Tensor* curr = weights[weight]->w;
		Tensor* m = weights[weight]->m;
		Tensor* v = weights[weight]->v;
		i32 elements = tensor_number_elements(curr);
		for (i32 i = 0; i < elements; ++i)
		{
			f32 grad = curr->grad->data[i];
			m->data[i] = p->b1 * m->data[i] + (1.0f - p->b1) * grad;
			v->data[i] = p->b2 * v->data[i] + (1.0f - p->b2) * grad * grad;

			f32 mhat = m->data[i] / (1.0f - powf(p->b1, t));
			f32 vhat = v->data[i] / (1.0f - powf(p->b2, t));
			curr->data[i] -= p->lr * mhat / (sqrtf(vhat) + p->eps);
		}
		tensor_fill(curr->grad, 0.0f);
	}
}
