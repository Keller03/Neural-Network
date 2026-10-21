

class NeuralNetwork:
    def __init__(self):
        # list which will contain the loss after training
        self.loss = []
        # the layer providing data
        self.data_layer = None
        # the layer calculating the loss and the prediction
        self.loss_layer = None
        # the definition of the particular neural network
        self.layers = []

    # TODO: the forward pass of the network during training
    # get data, iterate through the network layers and calculate the loss using the final loss layer
    def forward(self):
        #TODO replace pass with implementation
        input_tensor, target_tensor = self.data_layer.next()

        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        loss = self.loss_layer.forward(input_tensor, target_tensor)
        self.loss.append(loss)

        return loss

    # the backward pass during training - no need to change this at the moment
    def backward(self):
        pass

    # TODO: high level method to test a new input - return the prediction before the computation of the loss
    def test(self, input_tensor):
        #TODO replace pass with implementation
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        return input_tensor

    # TODO: append a new layer to the list of layers
    def append_layer(self, layer):
        #TODO replace pass with implementation
        self.layers.append(layer)
