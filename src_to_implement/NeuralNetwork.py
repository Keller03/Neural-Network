

import copy
from Optimization.Optimizers import Sgd


class NeuralNetwork:
    def __init__(self, learning_rate, weights_initializer=None, bias_initializer=None):
        # list which will contain the loss after training
        self.loss = []
        # the layer providing data
        self.data_layer = None
        # the layer calculating the loss and the prediction
        self.loss_layer = None
        # the definition of the particular neural network
        self.layers = []
        self.label_tensor = None
        self.learning_rate = learning_rate
        self.optimizer_class = Sgd(learning_rate)
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    # TODO: the forward pass of the network during training
    # get data, iterate through the network layers and calculate the loss using the final loss layer
    def forward(self):
        #TODO replace pass with implementation
        input_tensor, target_tensor = self.data_layer.next()
        self.label_tensor = target_tensor

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        loss = self.loss_layer.forward(input_tensor, target_tensor)
        self.loss.append(loss)

        return loss

    # the backward pass during training - no need to change this at the moment
    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in reversed(self.layers):
            error_tensor = layer.backward(error_tensor)
            

    # TODO: high level method to test a new input - return the prediction before the computation of the loss
    def test(self, input_tensor):
        #TODO replace pass with implementation
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    # TODO: append a new layer to the list of layers
    def append_layer(self, layer):
        if layer.trainable:

            if isinstance(self.optimizer_class, type):
                optimizer = self.optimizer_class(self.learning_rate)
            else:
                optimizer = copy.deepcopy(self.optimizer_class)

            layer.optimizer = optimizer

            if (
                self.weights_initializer is not None
                and self.bias_initializer is not None
            ):
                layer.initialize(
                    self.weights_initializer,
                    self.bias_initializer
                )

        self.layers.append(layer)

    def train(self, iterations):
        for _ in range(iterations):
            self.forward()
            self.backward()
