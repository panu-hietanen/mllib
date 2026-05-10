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