cmake_minimum_required(VERSION 3.20)
project(mllib C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_executable(mllib
    src/main.c
    src/arena.c
    src/tensor.c
    src/node.c
    src/graph.c
    src/optimizer.c
)

target_include_directories(mllib PRIVATE src)

if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(mllib PRIVATE /fsanitize=address)
endif()