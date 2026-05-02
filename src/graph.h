#ifndef GRAPH_H__
#define GRAPH_H__

#include "config.h"

#include "tensor.h"

void add_backward(const Tensor* t);
void matmul_backward(const Tensor* t);

void backward(const Tensor* t);

#endif // !GRAPH_H__
