#include "node.h"

Node* node_create(mem_arena* arena, const Tensor* a, const Tensor* b, void (*backward)(mem_arena* arena, const Tensor*), const void* aux)
{
	Node* new = PUSH_STRUCT(arena, Node);
	new->inputs[0] = a;
	new->inputs[1] = b;
	new->backward = backward;
	new->aux = aux;
	return new;
}
