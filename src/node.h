#ifndef NODE_H__
#define NODE_H__

typedef struct Tensor Tensor;

typedef struct {
	Tensor* inputs[2];
	void (*backward)(const Tensor*);
} Node;

#endif // !NODE_H__
