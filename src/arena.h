#ifndef ARENA_H__
#define ARENA_H__

#include "config.h"

typedef struct {
	u64 capacity;
	u64 pos;
} mem_arena;

mem_arena* arena_create(u64 size);
void arena_clear(mem_arena* arena);
void* arena_push(mem_arena* arena, u64 size, bool non_zero);
void arena_pop(mem_arena* arena, u64 size);
void arena_pop_to(mem_arena* arena, u64 pos);
void arena_destroy(mem_arena* arena);

#define ARENA_BASE  (sizeof(mem_arena))
#define ARENA_ALIGN (sizeof(void*))

#define PUSH_STRUCT(arena, T)			(T*)arena_push((arena), sizeof(T), false)
#define PUSH_STRUCT_NZ(arena, T)		(T*)arena_push((arena), sizeof(T), true)
#define PUSH_ARRAY(arena, T, n)			(T*)arena_push((arena), sizeof(T) * (n), false)
#define PUSH_ARRAY_NZ(arena, T, n)		(T*)arena_push((arena), sizeof(T) * (n), true)

#endif // !ARENA_H__
