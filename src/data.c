#include "data.h"

void data_save_weights(Tensor** weights, i32 n, const char* filename)
{}

void data_load_weights(Tensor** weights, i32 n, const char* filename)
{}

void data_export_csv(Tensor * x, Tensor * predictions, const char* filename)
{}

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
