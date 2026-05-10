import ctypes
import numpy as np
from numpy.typing import NDArray
from ._bindings import lib

from .tensor import Arena, Tensor

class Linear:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.ws = []
        self.ms = []
        self.vs = []

    def _init(self, arena: Arena):
        shape_w = [self.input_size, self.output_size]
        shape_b = [1, shape_w[1]]

        self.w = Tensor.xavier(arena, shape_w)
        self.mw, self.vw = Tensor.zeros(arena, shape_w), Tensor.zeros(arena, shape_w)

        self.ws.append(self.w)
        self.ms.append(self.mw)
        self.vs.append(self.vw)

        self.b = Tensor.zeros(arena, shape_b)
        self.mb, self.vb = Tensor.zeros(arena, shape_b), Tensor.zeros(arena, shape_b)

        self.ws.append(self.b)
        self.ms.append(self.mb)
        self.vs.append(self.vb)

    def _parameters(self) -> tuple[Tensor, Tensor, Tensor]:
        return self.ws, self.ms, self.vs
    
    def _forward(self, arena: Arena, x: Tensor) -> Tensor:
        return Tensor(lib.graph_add(arena._ptr, lib.graph_matmul(arena._ptr, x._ptr, self.w._ptr), self.b._ptr))
    
class ReLU:
    def _forward(self, arena: Arena, x: Tensor):
        return Tensor(lib.graph_relu(arena._ptr, x._ptr))
    

class Model:
    def __init__(self, layers, loss, lr, b1, b2, eps, arena_size):
        self.arena_p = Arena(arena_size)
        self.arena_t = Arena(arena_size)

        self.ws = []
        self.ms = []
        self.vs = []

        self.layers = layers
        for l in self.layers:
            if hasattr(l, "_init"):
                l._init(self.arena_p)

            if hasattr(l, "_parameters"):
                ws_l, ms_l, vs_l = l._parameters()
                self.ws.extend(ws_l)
                self.ms.extend(ms_l)
                self.vs.extend(vs_l)

        self.n_weights = len(self.ws)
        self._ws_arr = (ctypes.c_void_p * self.n_weights)(*[t._ptr for t in self.ws])
        self._ms_arr = (ctypes.c_void_p * self.n_weights)(*[t._ptr for t in self.ms])
        self._vs_arr = (ctypes.c_void_p * self.n_weights)(*[t._ptr for t in self.vs])

        self.loss = loss
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        
        self.n_step = 1

    def __del__(self):
        self.arena_p.destroy()
        self.arena_t.destroy()

    def forward(self, x: NDArray, target: NDArray) -> float:
        x_tensor = Tensor.from_numpy(self.arena_t, x)
        t_tensor = Tensor.from_numpy(self.arena_t, target)

        out: Tensor = x_tensor
        for l in self.layers:
            out = l._forward(self.arena_t, out)

        a_t = self.arena_t._ptr
        if self.loss == "softmax_ce":
            self._loss = Tensor(lib.graph_softmax_ce(a_t, out._ptr, t_tensor._ptr))
        elif self.loss == "sigmoid_bce":
            self._loss = Tensor(lib.graph_sigmoid_bce(a_t, out._ptr, t_tensor._ptr))
        elif self.loss == "mse":
            self._loss = Tensor(lib.graph_mse(a_t, out._ptr, t_tensor._ptr))
        elif self.loss == "ce":
            self._loss = Tensor(lib.graph_ce(a_t, out._ptr, t_tensor._ptr))
        else:
            raise ValueError("Loss function not recognised.")

        return self._loss.to_numpy()[0]
    
    def backward(self) -> None:
        lib.backward(self.arena_t._ptr, self._loss._ptr)

    def step(self) -> None:
        lib.adam_step_flat(self._ws_arr, self._ms_arr, self._vs_arr, self.n_weights,
                           self.b1, self.b2, self.eps, self.lr, self.n_step)
        self.n_step += 1
        self.arena_t.clear()
