from __future__ import annotations
import ctypes
import numpy as np
from numpy.typing import NDArray
from ._bindings import (
    arena_create, arena_clear, arena_destroy,
    tensor_zeros, tensor_xavier,
    tensor_set_data, tensor_get_data, tensor_number_elements,
)

class Arena:
    def __init__(self, size: int):
        self._ptr = arena_create(size)

    def __del__(self):
        self.destroy()

    def clear(self) -> None:
        arena_clear(self._ptr)

    def destroy(self) -> None:
        if self._ptr is not None:
            arena_destroy(self._ptr)
            self._ptr = None

class Tensor:
    def __init__(self, ptr: int):
        self._ptr = ptr

    @staticmethod
    def zeros(arena: Arena, shape: list[int]) -> Tensor:
        return Tensor(tensor_zeros(arena._ptr, shape))

    @staticmethod
    def xavier(arena: Arena, shape: list[int]) -> Tensor:
        return Tensor(tensor_xavier(arena._ptr, shape))

    @staticmethod
    def from_numpy(arena: Arena, array: NDArray) -> Tensor:
        t = Tensor.zeros(arena, list(array.shape))
        t.set_data(array)
        return t

    def set_data(self, numpy_array: NDArray) -> None:
        flat = numpy_array.flatten().astype("float32")
        buf = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        tensor_set_data(self._ptr, buf, flat.size)

    def to_numpy(self) -> NDArray:
        n = tensor_number_elements(self._ptr)
        result = np.zeros(n, dtype=np.float32)
        tensor_get_data(self._ptr, result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
        return result
