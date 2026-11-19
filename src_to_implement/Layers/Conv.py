import numpy as np
import copy
from Layers import Base

class Conv(Base.BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.trainable = True

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels

        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)

        self.gradient_weights = None
        self.gradient_bias = None

        self.optimizer = None
        self.bias_optimizer = None
        self.error_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        
        batch_size, channels, height, width = input_tensor.shape

        num_kernels, _, kh, kw = self.weights.shape

        sh, sw = self.stride_shape

        out_h = int(np.ceil(height / sh))
        out_w = int(np.ceil(width / sw))

        pad_h = max((out_h - 1) * sh + kh - height, 0)
        pad_w = max((out_w - 1) * sw + kw - width, 0)

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        self.padded_input = np.pad(
            input_tensor,
            (
                (0, 0),
                (0, 0),
                (pad_top, pad_bottom),
                (pad_left, pad_right)
            ),
            mode="constant"
        )

        output = np.zeros((batch_size, num_kernels, out_h, out_w))

        self.padding = (pad_top, pad_bottom, pad_left, pad_right)

        for n in range(batch_size):
            for k in range(num_kernels):
                for i in range(out_h):
                    for j in range(out_w):

                        h_start = i * sh
                        w_start = j * sw

                        patch = self.padded_input[
                            n,
                            :,
                            h_start:h_start + kh,
                            w_start:w_start + kw
                        ]

                        output[n, k, i, j] = (
                            np.sum(patch * self.weights[k])
                            + self.bias[k]
                        )

        return output
            

    def backward(self, error_tensor):

        batch_size, num_kernels, out_h, out_w = error_tensor.shape
        _, _, kh, kw = self.weights.shape

        sh, sw = self.stride_shape

        input_gradient = np.zeros_like(self.padded_input)

        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)

        for n in range(batch_size):
            for k in range(num_kernels):
                for i in range(out_h):
                    for j in range(out_w):

                        h_start = i * sh
                        w_start = j * sw

                        error = error_tensor[n, k, i, j]

                        input_patch = self.padded_input[
                            n, :,
                            h_start:h_start + kh,
                            w_start:w_start + kw
                        ]

                        input_gradient[
                            n, :,
                            h_start:h_start + kh,
                            w_start:w_start + kw
                        ] += self.weights[k] * error

                        self.gradient_weights[k] += input_patch * error

                        self.gradient_bias[k] += error

        pad_top, pad_bottom, pad_left, pad_right = self.padding

        h_end = -pad_bottom if pad_bottom > 0 else None
        w_end = -pad_right if pad_right > 0 else None

        input_gradient = input_gradient[
            :, :,
            pad_top:h_end,
            pad_left:w_end
        ]

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(
                self.weights, self.gradient_weights
            )

            self.bias = self.bias_optimizer.calculate_update(
                self.bias, self.gradient_bias
            )

        return input_gradient

    def initialize(self, weights_initializer, bias_initializer):

        fan_in = np.prod(self.convolution_shape)
        fan_out = self.num_kernels * np.prod(self.convolution_shape[1:])

        self.weights = weights_initializer.initialize(
            self.weights.shape,
            fan_in,
            fan_out
        )

        self.bias = bias_initializer.initialize(
            self.bias.shape,
            1,
            self.num_kernels
        )


    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)

    
        

