#ifndef NODE_H__
#define NODE_H__

#include "arena.h"

typedef struct Tensor Tensor;

typedef struct {
	Tensor* inputs[2];
	void (*backward)(mem_arena* arena, const Tensor*);
	void* aux;
} Node;

Node* node_create(mem_arena* arena, const Tensor* a, const Tensor* b, void (*backward)(mem_arena* arena, const Tensor*), const void* aux);

#endif // !NODE_H__
