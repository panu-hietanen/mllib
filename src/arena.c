#include "arena.h"

mem_arena* arena_create(u64 size)
{
	mem_arena* arena = malloc(size);
	if (!arena) exit(EXIT_FAILURE);

	arena->capacity = size;
	arena->pos = ARENA_BASE;

	return arena;
}

void arena_clear(mem_arena* arena)
{
	arena_pop_to(arena, ARENA_BASE);
}

void* arena_push(mem_arena* arena, u64 size, bool non_zero)
{
	u64 pos_aligned = ALIGN_UP_POW2(arena->pos, ARENA_ALIGN);
	u64 new_pos = pos_aligned + size;

	if (new_pos > arena->capacity) exit(EXIT_FAILURE);

	arena->pos = new_pos;

	u8* out = (u8*)arena + pos_aligned;
	if (!non_zero) memset(out, 0, size);

	return out;
}

void arena_pop(mem_arena* arena, u64 size)
{
	size = MIN(size, arena->pos - ARENA_BASE);
	arena->pos -= size;
}

void arena_pop_to(mem_arena* arena, u64 pos)
{
	u64 size = (pos < arena->pos) ? arena->pos - pos : 0;
	arena_pop(arena, size);
}

void arena_destroy(mem_arena* arena)
{
	free(arena);
}


