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

# Platform-specific options
if (MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(mllib_core PRIVATE /fsanitize=address)
elseif(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(mllib_core PRIVATE -fsanitize=address)
    target_link_options(mllib_core PUBLIC -fsanitize=address)
elseif(NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(mllib_core PRIVATE -march=native)
endif()

# Link math library on Linux
if (UNIX)
    target_link_libraries(mllib_core PUBLIC m)
endif()

find_package(BLAS REQUIRED)
target_link_libraries(mllib_core PUBLIC ${BLAS_LIBRARIES})

# Helper macro to add examples
macro(add_example name)
add_executable(${name} examples/${name}/main.c)
target_link_libraries(${name} PRIVATE mllib_core)
endmacro()

# Examples
add_example(spiral)
add_example(spiral_ce)
add_example(spiral_batch)
add_example(spiral_bce)
add_example(load_weights)

# Python bindings
add_library(mllib SHARED
src/arena.c
src/tensor.c
src/node.c
src/graph.c
src/optimizer.c
src/data.c
)
target_include_directories(mllib PUBLIC src)
if (UNIX)
target_link_libraries(mllib PUBLIC m)
endif()
if (NOT MSVC AND CMAKE_BUILD_TYPE STREQUAL "Release")
target_compile_options(mllib PRIVATE -march=native)
endif()

find_package(BLAS REQUIRED)
target_link_libraries(mllib PUBLIC ${BLAS_LIBRARIES})