#ifndef DATA_H__
#define DATA_H__

#include "config.h"

#include "arena.h"
#include "tensor.h"

void data_save_tensors(Tensor** weights, i32 n, const char* filename);
void data_load_weights(mem_arena* arena, Tensor** weights, i32 n, const char* filename);

void data_generate_spiral(i32 n, Tensor* data, Tensor* target);

#endif // !DATA_H__
