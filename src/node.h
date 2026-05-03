#ifndef NODE_H__
#define NODE_H__

#include "arena.h"

typedef struct Tensor Tensor;

typedef struct {
	Tensor* inputs[2];
	void (*backward)(mem_arena* arena, const Tensor*);
} Node;

#endif // !NODE_H__
