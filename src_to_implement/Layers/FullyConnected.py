import numpy as np
from Layers import Base

# TODO: Implement class FullyConnected
class FullyConnected(Base.BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (input_size + 1, output_size))
        self.input_tensor = None
        self.optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        bias = np.ones((input_tensor.shape[0], 1), dtype=input_tensor.dtype)
        input_with_bias = np.concatenate((input_tensor, bias), axis=1)
        return input_with_bias @ self.weights

    def backward(self, error_tensor):
        bias = np.ones((self.input_tensor.shape[0], 1), dtype=self.input_tensor.dtype)
        input_with_bias = np.concatenate((self.input_tensor, bias), axis=1)
        self.grad_weights = input_with_bias.T @ error_tensor

        input_gradient = error_tensor @ self.weights[:-1, :].T

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(
                self.weights, self.grad_weights
            )

        return input_gradient

    def initialize(self, weights_initializer, bias_initializer):

        self.weights[:-1, :] = weights_initializer.initialize(
            (self.input_size, self.output_size),
            self.input_size,
            self.output_size
        )

        self.weights[-1, :] = bias_initializer.initialize(
            (self.output_size,),
            1,
            self.output_size
        )
    
    @property
    def Optimizer(self):
        return self.optimizer

    @Optimizer.setter
    def Optimizer(self, optimizer):
        self.optimizer = optimizer

    @property
    def gradient_weights(self):
        return self.grad_weights

    

    

    

