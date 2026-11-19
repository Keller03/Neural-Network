import numpy as np

from src_to_implement.Layers import Base


class Pooling(Base.BaseLayer):
    def __init__(self, stride, pooling_shape):
        super().__init__()
        self.stride = stride
        self.pooling_shape = pooling_shape

        self.switches = None
        self.this_input_shape = None

    def forward(self, input_tensor):
        self.this_input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]

        # calculate the output shape
        self.out_shape = []
        for input_dim, pooling_dim, stride in zip(input_tensor.shape[2:], self.pooling_shape, self.stride):
            rounded = int(np.floor(float(input_dim - pooling_dim) / float(stride)) + 1)
            self.out_shape.append(rounded)

        self.out_shape = (self.this_input_shape[1], *self.out_shape)

        result = np.zeros((batch_size, *self.out_shape))
        self.switches = {}

        for this_batch in range(batch_size):
            for this_channel in range(self.out_shape[0]):
                for y_out in range(self.out_shape[1]):
                    y_in = self.stride[0] * y_out
                    for x_out in range(self.out_shape[2]):
                        x_in = self.stride[1] * x_out
                        input_part = input_tensor[
                                     this_batch,
                                     this_channel,
                                     y_in:y_in + self.pooling_shape[0],
                                     x_in:x_in + self.pooling_shape[1]]
                        max_idx = np.unravel_index(
                            input_part.argmax(),
                            input_part.shape)

                        shape = (this_batch, this_channel, y_out, x_out)
                        self.switches[shape] = \
                            (max_idx[0] + y_in, max_idx[1] + x_in)

                        result[shape] = input_part[max_idx]

        return result

    def backward(self, error_tensor):
        error_below = np.zeros(self.this_input_shape)

        for res_idx, max_idx in self.switches.items():
            shape = (res_idx[0], res_idx[1], *max_idx)
            error_below[shape] += error_tensor[res_idx]

        return error_below
