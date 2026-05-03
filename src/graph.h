#ifndef GRAPH_H__
#define GRAPH_H__

#include "config.h"

#include "arena.h"
#include "tensor.h"

Tensor* graph_add(mem_arena* arena, Tensor* a, Tensor* b);
Tensor* graph_matmul(mem_arena* arena, Tensor* a, Tensor* b);

void add_backward(mem_arena* arena, const Tensor* t);
void matmul_backward(mem_arena* arena, const Tensor* t);

void backward(mem_arena* arena, const Tensor* t);

#endif // !GRAPH_H__
