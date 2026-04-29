#ifndef TENSOR_H__
#define TENSOR_H__

#include "config.h"

#include "arena.h"

typedef struct {
	f32* data;
	int shape[MAX_DIMS];
	int ndim;
} Tensor;

Tensor* tensor_create(mem_arena* arena, int* shape, int ndim, bool non_zero);
f32* tensor_at(const Tensor* t, int* indices);
void tensor_print(const Tensor* t);

int tensor_number_elements(const Tensor* t);

void tensor_fill(Tensor* t, f32 val);
Tensor* tensor_copy(mem_arena* arena, const Tensor* t);
Tensor* tensor_zeros(mem_arena* arena, int* shape, int ndim);
Tensor* tensor_ones(mem_arena* arena, int* shape, int ndim);

#endif // !TENSOR_H__
