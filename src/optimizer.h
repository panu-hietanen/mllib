#ifndef OPTIMIZER_H__
#define OPTIMIZER_H__

#include "config.h"

#include "tensor.h"

typedef struct {
	f32 b1;
	f32 b2;
	f32 eps;
	f32 lr;
} OptimParams;

typedef struct {
	Tensor* w;
	Tensor* m;
	Tensor* v;
} AdamWeight;

void sgd_step(Tensor** weights, i32 n, OptimParams* p);
void adam_step(AdamWeight** weights, i32 n, const OptimParams* p, i32 t);
void adam_step_flat(Tensor** ws, Tensor** ms, Tensor** vs, i32 n, f32 b1, f32 b2, f32 eps, f32 lr, i32 t);

#endif // !OPTIMIZER_H__
