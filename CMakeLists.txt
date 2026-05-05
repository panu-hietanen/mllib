cmake_minimum_required(VERSION 3.20)
project(mllib C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Core library
add_library(mllib_core
    src/arena.c
    src/tensor.c
    src/node.c
    src/graph.c
    src/optimizer.c
    src/data.c
)

target_include_directories(mllib_core PUBLIC src)

if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(mllib_core PRIVATE /fsanitize=address)
endif()

# Helper macro to add examples
macro(add_example name)
    add_executable(${name} examples/${name}/main.c)
    target_link_libraries(${name} PRIVATE mllib_core)
endmacro()

# Examples
add_example(spiral)
