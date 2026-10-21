from Layers.Base import *
import numpy as np

# TODO: Implement class SoftMax
class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        input_tensor_stable = input_tensor - np.max(input_tensor, axis=1, keepdims=True)
        exp_values = np.exp(input_tensor_stable)
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output_tensor = probabilities
        return probabilities

    def backward(self, error_tensor):
        return None

    