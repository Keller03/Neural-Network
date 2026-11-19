import numpy as np

import NeuralNetwork
import Helpers3
from Optimization import Initializers, Optimizers, Loss
from Layers import Conv, FullyConnected, ReLU, Flatten, SoftMax
from src_to_implement.Optimization import Pooling

def get_network():
    net = NeuralNetwork.NeuralNetwork(1e-4, Initializers.He(), Initializers.Constant(0.1))
    net.optimizer_class = Optimizers.Sgd
    input_image_shape = (1, 8, 8)
    conv_stride_shape = (1, 1)
    convolution_shape = (1, 3, 3)
    categories = 10
    num_kernels = 4

    cl_1 = Conv.Conv(conv_stride_shape, convolution_shape, num_kernels)
    net.append_layer(cl_1)
    cl_1_output_shape = (*input_image_shape[1:], num_kernels)
    net.append_layer(ReLU.ReLU())

    pool = Pooling.Pooling((2, 2), (2, 2))
    pool_output_shape = (4, 4, 4)
    net.append_layer(pool)
    fcl_1_input_size = np.prod(pool_output_shape)

    net.append_layer(Flatten.Flatten())

    fcl_1 = FullyConnected.FullyConnected(fcl_1_input_size, int(fcl_1_input_size/2.))
    net.append_layer(fcl_1)

    net.append_layer(ReLU.ReLU())

    fcl_2 = FullyConnected.FullyConnected(int(fcl_1_input_size/2.), categories)
    net.append_layer(fcl_2)

    net.append_layer(SoftMax.SoftMax())



if __name__ == '__main__':
    network = get_network()

    # TODO: Add some functionality to plot examples from the training data with their labels. Include an overview figure in your report.

    network.data_layer = Helpers3.DigitData(batch_size)
    network.loss_layer = Loss.CrossEntropyLoss()

    # TODO: Add some functionality to plot the training curves as well as some exemplary filter kernels before and after training
    # Include an overview figure in your written report

    network.train(200)


    data, labels = network.data_layer.get_test_set()

    results = network.test(data)

    # TODO: Provide a small statistic over 10 training runs of your network (with new initialization). Report the mean and standard deviation over these runs.
    # Bonus: If you want, adapt the training to experiment with certain parameters (e.g., additional filters, more layers, different learning rate)
    # Note that this network is still somewhat instable, so you might run into instabilities.
    accuracy = Helpers3.calculate_accuracy(results, labels)

    print('\nOn the UCI ML hand-written digits dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
