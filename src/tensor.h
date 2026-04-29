#ifndef TENSOR_H__
#define TENSOR_H__

#include "config.h"

#include "arena.h"

typedef struct {
	f32* data;
	i32 shape[MAX_DIMS];
	i32 ndim;
} Tensor;

Tensor* tensor_create(mem_arena* arena, const i32* shape, i32 ndim, bool non_zero);
f32* tensor_at(const Tensor* t, i32* indices);
void tensor_print(const Tensor* t);

i32 tensor_number_elements(const Tensor* t);

void tensor_fill(Tensor* t, f32 val);
Tensor* tensor_copy(mem_arena* arena, const Tensor* t);
Tensor* tensor_zeros(mem_arena* arena, i32* shape, i32 ndim);
Tensor* tensor_ones(mem_arena* arena, i32* shape, i32 ndim);

#endif // !TENSOR_H__
