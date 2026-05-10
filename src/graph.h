#ifndef GRAPH_H__
#define GRAPH_H__

#include "config.h"

#include "arena.h"
#include "tensor.h"

// Ops
Tensor* graph_add(mem_arena* arena, Tensor* a, Tensor* b);
Tensor* graph_matmul(mem_arena* arena, Tensor* a, Tensor* b);

// Nonlinearities
Tensor* graph_relu(mem_arena* arena, Tensor* a);
Tensor* graph_softmax(mem_arena* arena, Tensor* a);
Tensor* graph_sigmoid(mem_arena* arena, Tensor* a);

// Loss functions
Tensor* graph_mse(mem_arena* arena, Tensor* a, Tensor* b);
Tensor* graph_ce(mem_arena* arena, Tensor* a, Tensor* b);

// Fused
Tensor* graph_softmax_ce(mem_arena* arena, Tensor* a, Tensor* b);
Tensor* graph_sigmoid_bce(mem_arena* arena, Tensor* a, Tensor* b);

// Autodiff
void add_backward(mem_arena* arena, const Tensor* t);
void matmul_backward(mem_arena* arena, const Tensor* t);
void relu_backward(mem_arena* arena, const Tensor* t);
void mse_backward(mem_arena* arena, const Tensor* t);
void softmax_backward(mem_arena* arena, const Tensor* t);
void ce_backward(mem_arena* arena, const Tensor* t);
void softmax_ce_backward(mem_arena* arena, const Tensor* t);
void sigmoid_backward(mem_arena* arena, const Tensor* t);
void sigmoid_bce_backward(mem_arena* arena, const Tensor* t);

// Operate on graph
i32 visit(Tensor** visited_list, i32 n, Tensor* t);
void backward(mem_arena* arena, Tensor* t);

#endif // !GRAPH_H__
