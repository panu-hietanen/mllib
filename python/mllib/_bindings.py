import ctypes
from pathlib import Path

_lib_path = Path(__file__).parent.parent.parent / "out/build/linux-release/libmllib.so"
lib = ctypes.CDLL(str(_lib_path))

# ARENA
lib.arena_create.argtypes = [ctypes.c_uint64]
lib.arena_create.restype = ctypes.c_void_p

lib.arena_clear.argtypes = [ctypes.c_void_p]
lib.arena_clear.restype = None

lib.arena_destroy.argtypes = [ctypes.c_void_p]
lib.arena_destroy.restype = None

# TENSOR
lib.tensor_create.argtypes = [ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_int),
                               ctypes.c_int,
                               ctypes.c_bool]
lib.tensor_create.restype = ctypes.c_void_p

lib.tensor_zeros.argtypes = [ctypes.c_void_p,
                              ctypes.POINTER(ctypes.c_int),
                              ctypes.c_int]
lib.tensor_zeros.restype = ctypes.c_void_p

lib.tensor_ones.argtypes = [ctypes.c_void_p,
                             ctypes.POINTER(ctypes.c_int),
                             ctypes.c_int]
lib.tensor_ones.restype = ctypes.c_void_p

lib.tensor_xavier.argtypes = [ctypes.c_void_p,
                               ctypes.POINTER(ctypes.c_int),
                               ctypes.c_int]
lib.tensor_xavier.restype = ctypes.c_void_p

lib.tensor_fill.argtypes = [ctypes.c_void_p, ctypes.c_float]
lib.tensor_fill.restype = None

lib.tensor_set_data.argtypes = [ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.c_int]
lib.tensor_set_data.restype = None

lib.tensor_get_data.argtypes = [ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_float),
                                 ctypes.c_int]
lib.tensor_get_data.restype = None

lib.tensor_number_elements.argtypes = [ctypes.c_void_p]
lib.tensor_number_elements.restype = ctypes.c_int

# GRAPH
lib.graph_add.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.graph_add.restype = ctypes.c_void_p

lib.graph_matmul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.graph_matmul.restype = ctypes.c_void_p

lib.graph_relu.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.graph_relu.restype = ctypes.c_void_p

lib.graph_sigmoid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.graph_sigmoid.restype = ctypes.c_void_p

lib.graph_softmax_ce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.graph_softmax_ce.restype = ctypes.c_void_p

lib.graph_sigmoid_bce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.graph_sigmoid_bce.restype = ctypes.c_void_p

lib.graph_mse.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.graph_mse.restype = ctypes.c_void_p

lib.graph_ce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
lib.graph_ce.restype = ctypes.c_void_p

lib.backward.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib.backward.restype = None

# OPTIMIZER
lib.adam_step_flat.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                ctypes.POINTER(ctypes.c_void_p),
                                ctypes.POINTER(ctypes.c_void_p),
                                ctypes.c_int,
                                ctypes.c_float,
                                ctypes.c_float,
                                ctypes.c_float,
                                ctypes.c_float,
                                ctypes.c_int]
lib.adam_step_flat.restype = None

# IO
lib.data_save_tensors.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.c_int,
                                   ctypes.c_char_p]
lib.data_save_tensors.restype = None

lib.data_load_weights.argtypes = [ctypes.c_void_p,
                                   ctypes.POINTER(ctypes.c_void_p),
                                   ctypes.c_int,
                                   ctypes.c_char_p]
lib.data_load_weights.restype = None

########################################################################### Typed wrappers

# ARENA
def arena_create(size: int) -> int:
    return lib.arena_create(size)

def arena_clear(arena: int) -> None:
    lib.arena_clear(arena)

def arena_destroy(arena: int) -> None:
    lib.arena_destroy(arena)

# TENSOR
def tensor_zeros(arena: int, shape: list[int]) -> int:
    arr = (ctypes.c_int * len(shape))(*shape)
    return lib.tensor_zeros(arena, arr, len(shape))

def tensor_ones(arena: int, shape: list[int]) -> int:
    arr = (ctypes.c_int * len(shape))(*shape)
    return lib.tensor_ones(arena, arr, len(shape))

def tensor_xavier(arena: int, shape: list[int]) -> int:
    arr = (ctypes.c_int * len(shape))(*shape)
    return lib.tensor_xavier(arena, arr, len(shape))

def tensor_fill(tensor: int, val: float) -> None:
    lib.tensor_fill(tensor, val)

def tensor_set_data(tensor: int, data: ctypes._Pointer[ctypes.c_float], n: int) -> None:
    lib.tensor_set_data(tensor, data, n)

def tensor_get_data(tensor: int, data: ctypes._Pointer[ctypes.c_float], n: int) -> None:
    lib.tensor_get_data(tensor, data, n)

def tensor_number_elements(tensor: int) -> int:
    return lib.tensor_number_elements(tensor)

# GRAPH
def graph_add(arena: int, a: int, b: int) -> int:
    return lib.graph_add(arena, a, b)

def graph_matmul(arena: int, a: int, b: int) -> int:
    return lib.graph_matmul(arena, a, b)

def graph_relu(arena: int, a: int) -> int:
    return lib.graph_relu(arena, a)

def graph_sigmoid(arena: int, a: int) -> int:
    return lib.graph_sigmoid(arena, a)

def graph_softmax_ce(arena: int, a: int, b: int) -> int:
    return lib.graph_softmax_ce(arena, a, b)

def graph_sigmoid_bce(arena: int, a: int, b: int) -> int:
    return lib.graph_sigmoid_bce(arena, a, b)

def graph_mse(arena: int, a: int, b: int) -> int:
    return lib.graph_mse(arena, a, b)

def graph_ce(arena: int, a: int, b: int) -> int:
    return lib.graph_ce(arena, a, b)

def backward(arena: int, tensor: int) -> None:
    lib.backward(arena, tensor)

# OPTIMIZER
def adam_step_flat(ws: ctypes.Array, ms: ctypes.Array, vs: ctypes.Array,
                   n: int, b1: float, b2: float, eps: float, lr: float, t: int) -> None:
    lib.adam_step_flat(ws, ms, vs, n, b1, b2, eps, lr, t)

# IO
def data_save_tensors(tensors: ctypes.Array, n: int, path: str) -> None:
    lib.data_save_tensors(tensors, n, path.encode())

def data_load_weights(arena: int, out: ctypes.Array, n: int, path: str) -> None:
    lib.data_load_weights(arena, out, n, path.encode())
