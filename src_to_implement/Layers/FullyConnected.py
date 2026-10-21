import numpy as np
from Layers.Base import *

# TODO: Implement class FullyConnected
class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self._optimizer = None
        self._gradient_weights = None
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        bias = np.ones((input_tensor.shape[0], 1), dtype=input_tensor.dtype)
        input_with_bias = np.concatenate((input_tensor, bias), axis=1)
        return input_with_bias @ self.weights

    def backward(self, error_tensor):
        return None

    

    

