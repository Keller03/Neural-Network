from Layers import Base

class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        return input_tensor.reshape(input_tensor.shape[0], -1)

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)