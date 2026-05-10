import ctypes
import numpy as np
import sys
# sys.path.insert(0, "python")

print(sys.path)

from mllib._bindings import lib
from mllib.tensor import Arena, Tensor

arena = Arena(1024 * 1024)


data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
t = Tensor.from_numpy(arena, data)

result = t.to_numpy()
print("n elements:", lib.tensor_number_elements(t._ptr))
print("Input: ", data.flatten())
print("Output: ", result)
print("Match: ", np.allclose(data.flatten(), result))

arena.destroy()