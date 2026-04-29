#include "config.h"

#include "arena.h"

int main()
{
	mem_arena arena = arena_create(MiB(1));

	arena_destroy(arena);

	return 0;
}