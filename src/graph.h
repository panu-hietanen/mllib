#ifndef GRAPH_H__
#define GRAPH_H__

#include "config.h"

#include "arena.h"
#include "tensor.h"

void add_backward(mem_arena* arena, const Tensor* t);
void matmul_backward(mem_arena* arena, const Tensor* t);

void backward(const Tensor* t);

#endif // !GRAPH_H__
