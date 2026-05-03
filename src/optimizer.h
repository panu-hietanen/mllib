#ifndef OPTIMIZER_H__
#define OPTIMIZER_H__

#include "config.h"

#include "tensor.h"

void step(Tensor** weights, i32 n, f32 lr);

#endif // !OPTIMIZER_H__
