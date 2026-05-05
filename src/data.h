#ifndef DATA_H__
#define DATA_H__

#include "config.h"

#include "arena.h"
#include "tensor.h"

void data_save_weights(Tensor** weights, i32 n, const char* filename);
void data_load_weights(Tensor** weights, i32 n, const char* filename);
void data_export_csv(Tensor* x, Tensor* predictions, const char* filename);

void data_generate_spiral(i32 n, Tensor* data, Tensor* target);

#endif // !DATA_H__
