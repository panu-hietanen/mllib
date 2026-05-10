import ctypes
import numpy as np
from numpy.typing import NDArray
from ._bindings import (
    graph_add, graph_matmul, graph_relu, graph_sigmoid,
    graph_softmax_ce, graph_sigmoid_bce, graph_mse, graph_ce,
    adam_step_flat, backward,
    data_save_tensors,
)
from .tensor import Arena, Tensor


class Linear:
    def __init__(self, input_size: int, output_size: int):
        self.input_size = input_size
        self.output_size = output_size
        self.ws: list[Tensor] = []
        self.ms: list[Tensor] = []
        self.vs: list[Tensor] = []

    def _init(self, arena: Arena) -> None:
        shape_w = [self.input_size, self.output_size]
        shape_b = [1, self.output_size]

        self.w  = Tensor.xavier(arena, shape_w)
        self.mw = Tensor.zeros(arena, shape_w)
        self.vw = Tensor.zeros(arena, shape_w)

        self.b  = Tensor.zeros(arena, shape_b)
        self.mb = Tensor.zeros(arena, shape_b)
        self.vb = Tensor.zeros(arena, shape_b)

        self.ws.extend([self.w, self.b])
        self.ms.extend([self.mw, self.mb])
        self.vs.extend([self.vw, self.vb])

    def _parameters(self) -> tuple[list[Tensor], list[Tensor], list[Tensor]]:
        return self.ws, self.ms, self.vs

    def _forward(self, arena: Arena, x: Tensor) -> Tensor:
        return Tensor(graph_add(arena._ptr,
                          graph_matmul(arena._ptr, x._ptr, self.w._ptr),
                          self.b._ptr))


class ReLU:
    def _forward(self, arena: Arena, x: Tensor) -> Tensor:
        return Tensor(graph_relu(arena._ptr, x._ptr))


class Sigmoid:
    def _forward(self, arena: Arena, x: Tensor) -> Tensor:
        return Tensor(graph_sigmoid(arena._ptr, x._ptr))


class Model:
    def __init__(self, layers: list, loss: str, lr: float,
                 b1: float = 0.9, b2: float = 0.999,
                 eps: float = 1e-8, arena_size: int = 64 * 1024 * 1024):
        self.arena_p = Arena(arena_size)
        self.arena_t = Arena(arena_size)

        self.layers = layers
        self.ws: list[Tensor] = []
        self.ms: list[Tensor] = []
        self.vs: list[Tensor] = []

        for layer in self.layers:
            if hasattr(layer, "_init"):
                layer._init(self.arena_p)
            if hasattr(layer, "_parameters"):
                ws_l, ms_l, vs_l = layer._parameters()
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
        self._loss: Tensor | None = None

    def __del__(self) -> None:
        self.arena_p.destroy()
        self.arena_t.destroy()

    def forward(self, x: NDArray, target: NDArray) -> float:
        x_tensor = Tensor.from_numpy(self.arena_t, x)
        t_tensor = Tensor.from_numpy(self.arena_t, target)

        out = x_tensor
        for layer in self.layers:
            out = layer._forward(self.arena_t, out)

        a_t = self.arena_t._ptr
        if self.loss == "softmax_ce":
            self._loss = Tensor(graph_softmax_ce(a_t, out._ptr, t_tensor._ptr))
        elif self.loss == "sigmoid_bce":
            self._loss = Tensor(graph_sigmoid_bce(a_t, out._ptr, t_tensor._ptr))
        elif self.loss == "mse":
            self._loss = Tensor(graph_mse(a_t, out._ptr, t_tensor._ptr))
        elif self.loss == "ce":
            self._loss = Tensor(graph_ce(a_t, out._ptr, t_tensor._ptr))
        else:
            raise ValueError(f"Loss function '{self.loss}' not recognised.")

        return float(self._loss.to_numpy()[0])

    def backward(self) -> None:
        if self._loss is None:
            raise RuntimeError("Call forward() before backward().")
        backward(self.arena_t._ptr, self._loss._ptr)

    def step(self) -> None:
        adam_step_flat(self._ws_arr, self._ms_arr, self._vs_arr,
                       self.n_weights, self.b1, self.b2, self.eps, self.lr, self.n_step)
        self.n_step += 1
        self.arena_t.clear()

    def predict(self, x: NDArray) -> NDArray:
        x_tensor = Tensor.from_numpy(self.arena_t, x)
        out = x_tensor
        for layer in self.layers:
            out = layer._forward(self.arena_t, out)
        result = out.to_numpy()
        self.arena_t.clear()
        return result

    def save(self, path: str) -> None:
        arr = (ctypes.c_void_p * self.n_weights)(*[t._ptr for t in self.ws])
        data_save_tensors(arr, self.n_weights, path)
