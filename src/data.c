#include "data.h"

void data_save_tensors(Tensor** weights, i32 n, const char* filename)
{
	FILE* fptr;
	fptr = fopen(filename, "w");
	assert(fptr != NULL);
	for (i32 w = 0; w < n; ++w)
	{
		Tensor* weight = weights[w];
		i32 elements = tensor_number_elements(weight);
		fprintf(fptr, "%d,", weight->ndim);
		for (i32 i = 0; i < weight->ndim; ++i)
		{
			fprintf(fptr, "%d,", weight->shape[i]);
		}
		for (i32 i = 0; i < elements; ++i)
		{
			fprintf(fptr, "%.9g", weight->data[i]);
			if (i != elements - 1) 
			{
				fprintf(fptr, ",");
			}
			else
			{
				fprintf(fptr, "\n");
			}
		}
	}
	fclose(fptr);
}

void data_load_weights(mem_arena* arena, Tensor** out, i32 n, const char* filename)
{
	FILE* fptr;
	fptr = fopen(filename, "r");
	assert(fptr != NULL);

	for (i32 w = 0; w < n; ++w)
	{
		i32 ndim;
		fscanf(fptr, "%d,", &ndim);

		i32 shape[MAX_DIMS];
		for (i32 i = 0; i < ndim; ++i)
		{
			fscanf(fptr, "%d,", &shape[i]);
		}

		Tensor* weight = tensor_create(arena, shape, ndim, true);
		i32 elements = tensor_number_elements(weight);
		for (i32 i = 0; i < elements; ++i)
		{
			fscanf(fptr, "%g", &weight->data[i]);
			if (i != elements - 1) fscanf(fptr, ",");
		}
		fscanf(fptr, "\n");
		out[w] = weight;
	}
}

void data_generate_spiral(i32 n, Tensor* data, Tensor* target)
{
	f32 offset[2] = { 0, M_PI };
	for (i32 i = 0; i < n; ++i)
	{
		f32 t = 4.0f * (f32)M_PI * (f32)i / (f32)n;
		for (i32 j = 0; j < 2; ++j)
		{
			f32 noise = (((f32)rand() / (f32)RAND_MAX) * 2.0f - 1.0f) * 0.2f;
			f32 x = t * cosf(t + offset[j]) + noise;
			f32 y = t * sinf(t + offset[j]) + noise;

			data->data  [(i + (n * j)) * 2    ] = x;
			data->data  [(i + (n * j)) * 2 + 1] = y;

			target->data[i + n * j] = j;
		}
	}
}
