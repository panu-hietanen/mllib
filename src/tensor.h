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

#endif // !TENSOR_H__
