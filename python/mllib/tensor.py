import ctypes
import numpy as np
from numpy.typing import NDArray
from ._bindings import lib

class Arena:
    def __init__(self, size):
        self._ptr = lib.arena_create(size)

    def __del__(self):
        self.destroy()
    
    def clear(self):
        lib.arena_clear(self._ptr)

    def destroy(self):
        if self._ptr is not None:
            lib.arena_destroy(self._ptr)
            self._ptr = None

class Tensor:
    def __init__(self, ptr):
        self._ptr = ptr

    @staticmethod
    def zeros(arena: Arena, shape):
        arr = (ctypes.c_int * len(shape))(*shape)
        ptr = lib.tensor_zeros(arena._ptr, arr, len(shape))
        return Tensor(ptr)
    
    @staticmethod
    def xavier(arena: Arena, shape):
        arr = (ctypes.c_int * len(shape))(*shape)
        ptr = lib.tensor_xavier(arena._ptr, arr, len(shape))
        return Tensor(ptr)
    
    @staticmethod 
    def from_numpy(arena: Arena, array: NDArray):
        t = Tensor.zeros(arena, list(array.shape))
        t.set_data(array)
        return t

    def set_data(self, numpy_array: NDArray):
        flat = numpy_array.flatten().astype("float32")
        buf = flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        lib.tensor_set_data(self._ptr, buf, flat.size)

    def to_numpy(self) -> NDArray:
        n = lib.tensor_number_elements(self._ptr)
        result = np.zeros(n, dtype=np.float32)
        lib.tensor_get_data(self._ptr, result.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), n)
        return result