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
	memcpy(t->shape, shape, ndim * sizeof(i32));
	t->ndim = ndim;
	t->visited = false;

	t->grad = arena_push(arena, sizeof(Tensor), true);
	t->grad->data = arena_push(arena, size * sizeof(f32), false);
	memcpy(t->grad->shape, shape, ndim * sizeof(i32));
	t->grad->ndim = ndim;

	t->node = NULL;
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

Tensor *tensor_gather(mem_arena *arena, const Tensor *t, const i32 *indices, i32 n)
{
	assert(t->ndim == 2);

	i32 rows = n;
	i32 cols = t->shape[1];
	i32 shape[MAX_DIMS] = { rows, cols };

	Tensor* new = tensor_create(arena, shape, t->ndim, true);
	for (i32 r = 0; r < rows; ++r)
	{
		memcpy(new->data + r * cols, t->data + indices[r] * cols, cols * sizeof(f32));
	}
	return new;
}

void tensor_set_data(Tensor *t, const f32 *data, i32 n)
{
	assert(tensor_number_elements(t) == n);
	memcpy(t->data, data, n * sizeof(f32));
}

void tensor_get_data(const Tensor *t, f32 *data, i32 n)
{
	assert(tensor_number_elements(t) == n);
	memcpy(data, t->data, n * sizeof(f32));
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
	Tensor* new = tensor_create(arena, shape, ndim, false);
	return new;
}

Tensor* tensor_ones(mem_arena* arena, i32* shape, i32 ndim)
{
	Tensor* new = tensor_create(arena, shape, ndim, true);
	tensor_fill(new, 1.0);
	return new;
}

Tensor* tensor_rand(mem_arena* arena, i32* shape, i32 ndim)
{
	Tensor* new = tensor_create(arena, shape, ndim, true);
	i32 elements = tensor_number_elements(new);
	for (i32 i = 0; i < elements; ++i)
	{
		new->data[i] = ((f32)rand() / (f32)RAND_MAX) * 2.0f - 1.0f;
	}
	return new;
}

Tensor* tensor_xavier(mem_arena* arena, i32* shape, i32 ndim)
{
	Tensor* new = tensor_create(arena, shape, ndim, true);
	f32 scale = 1.0f / sqrtf((f32)shape[0]);
	i32 elements = tensor_number_elements(new);
	for (i32 i = 0; i < elements; ++i)
	{
		new->data[i] = scale * (((f32)rand() / (f32)RAND_MAX) * 2.0f - 1.0f);
	}
	return new;
}

Tensor* tensor_trans(mem_arena* arena, const Tensor* a)
{
	assert(a->ndim == 2);

	i32 m = a->shape[0];
	i32 n = a->shape[1];

	i32 shape[MAX_DIMS] = { n, m };

	Tensor* new = tensor_create(arena, shape, a->ndim, true);
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < m; ++i)
	{
		for (i32 j = 0; j < n; ++j)
		{
			new->data[j * m + i] = a->data[i * n + j];
		}
	}
	return new;
}

Tensor* tensor_add(mem_arena* arena, const Tensor* a, const Tensor* b)
{
	assert(a->ndim == b->ndim);
	i32 shape_new[MAX_DIMS];
	for (i32 i = 0; i < a->ndim; ++i)
	{
		assert(a->shape[i] == b->shape[i] || a->shape[i] == 1 || b->shape[i] == 1);
		shape_new[i] = MAX(a->shape[i], b->shape[i]);
	}
	Tensor* new = tensor_create(arena, shape_new, a->ndim, true);

	i32 elements = tensor_number_elements(new);
	i32 out_rows = shape_new[0];
	i32 out_cols = shape_new[1];
	for (i32 i = 0; i < elements; ++i)
	{
		i32 r = i / out_cols;
		i32 c = i % out_cols;

		i32 aidx = (a->shape[0] == 1 ? 0 : r) * a->shape[1] + (a->shape[1] == 1 ? 0 : c);
		i32 bidx = (b->shape[0] == 1 ? 0 : r) * b->shape[1] + (b->shape[1] == 1 ? 0 : c);

		new->data[i] = a->data[aidx] + b->data[bidx];
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

	i32 M = a->shape[0];
	i32 K = a->shape[1];
	i32 N = b->shape[1];

	i32 shape[MAX_DIMS] = { M, N };
	Tensor* new = tensor_create(arena, shape, a->ndim, false);
	for (i32 ii = 0; ii < M; ii += TILE)
	{
		for (i32 kk = 0; kk < K; kk += TILE)
		{
			for (i32 jj = 0; jj < N; jj += TILE)
			{
				for (i32 i = ii; i < MIN(ii+TILE, M); ++i)
				{
					f32* restrict       out_row = new->data + i * N;
					const f32* restrict a_row   = a->data   + i * K;
					for (i32 k = kk; k < MIN(kk+TILE, K); ++k)
					{
						f32 a_ik = a_row[k];
						const f32* restrict b_row = b->data + k * N;
						for (i32 j = jj; j < MIN(jj+TILE, N); ++j)
						{
							out_row[j] += a_ik * b_row[j];
						}
					}
				}

			}
		}
	}

	return new;
}

Tensor* tensor_relu(mem_arena* arena, const Tensor* a)
{
	Tensor* new = tensor_create(arena, a->shape, a->ndim, true);
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		new->data[i] = (a->data[i] > 0) ? a->data[i] : 0;
	}
	return new;
}

Tensor* tensor_mse(mem_arena* arena, const Tensor* a, const Tensor* b)
{
	assert(a->ndim == 2 && a->ndim == b->ndim);
	assert(a->shape[1] == 1);
	Tensor* new = tensor_create(arena, (i32[]) { 1 }, 1, false);
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		new->data[0] += (a->data[i] - b->data[i]) * (a->data[i] - b->data[i]);
	}
	new->data[0] /= elements;
	return new;
}

Tensor* tensor_ce(mem_arena* arena, const Tensor* a, const Tensor* b)
{
	assert(a->ndim == 2 && a->ndim == b->ndim);
	for (i32 i = 0; i < a->ndim; ++i)
		assert(a->shape[i] == b->shape[i]);
	i32 a_rows = a->shape[0];
	i32 a_cols = a->shape[1];

	Tensor* new = tensor_create(arena, (i32[]) { 1 }, 1, false);
	f32 loss = 0.0f;
	for (i32 r = 0; r < a_rows; ++r)
	{
		for (i32 c = 0; c < a_cols; ++c)
		{
			i32 idx = r * a_cols + c;
			loss -= b->data[idx] * logf(fmaxf(a->data[idx], 1e-7f));
		}
	}
	new->data[0] = loss / a_rows;
	return new;
}

Tensor *tensor_bce(mem_arena *arena, const Tensor *a, const Tensor *b)
{
	assert(a->ndim == 2 && a->shape[1] == 1 && b->ndim == a->ndim);
	for (i32 i = 0; i < a->ndim; ++i)
		assert(a->shape[i] == b->shape[i]);
	i32 a_rows = a->shape[0];
	i32 a_cols = a->shape[1];

	Tensor* new = tensor_create(arena, (i32[]) { 1 }, 1, false);
	f32 loss = 0.0f;
	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		f32 y = b->data[i];
		f32 p = fminf(0.999f, fmaxf(a->data[i], 1e-7f));
		loss -= y * logf(p) + (1 - y) * logf(1 - p);
	}
	new->data[0] = loss / elements;
	return new;
}

Tensor* tensor_softmax(mem_arena* arena, const Tensor* a)
{
	assert(a->ndim == 2);

	Tensor* new = tensor_create(arena, a->shape, a->ndim, true);

	i32 a_rows = a->shape[0];
	i32 a_cols = a->shape[1];

	f32* max_val = PUSH_ARRAY(arena, f32, a_rows);
	for (i32 r = 0; r < a_rows; ++r)
	{
		for (i32 c = 0; c < a_cols; ++c)
		{
			i32 idx = r * a_cols + c;
			if (c == 0)
			{
				max_val[r] = a->data[idx];
				continue;
			}
			if (a->data[idx] > max_val[r]) max_val[r] = a->data[idx];
		}
	}

	f32* denom = PUSH_ARRAY(arena, f32, a_rows);
	for (i32 r = 0; r < a_rows; ++r)
	{
		for (i32 c = 0; c < a_cols; ++c)
		{
			i32 idx = r * a_cols + c;
			if (c == 0)
			{
				denom[r] = 0.0f;
			}
			denom[r] += expf(a->data[idx] - max_val[r]);
		}
	}
	for (i32 r = 0; r < a_rows; ++r)
	{
		for (i32 c = 0; c < a_cols; ++c)
		{
			i32 idx = r * a_cols + c;
			new->data[idx] = expf(a->data[idx] - max_val[r]) / denom[r];
		}
	}
	return new;
}

Tensor *tensor_sigmoid(mem_arena *arena, const Tensor *a)
{
	assert(a->ndim == 2);

	Tensor *new = tensor_create(arena, a->shape, a->ndim, true);

	i32 elements = tensor_number_elements(a);
	for (i32 i = 0; i < elements; ++i)
	{
		new->data[i] = 1 / (1 + expf(-a->data[i]));
	}
	return new;
}
