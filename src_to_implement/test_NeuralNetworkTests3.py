import unittest
from Layers import FullyConnected, ReLU, SoftMax, Flatten, Conv, Dropout, Base
from Optimization import Loss, Optimizers, Initializers
import numpy as np
import NeuralNetwork
import Helpers3
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import os
from pathlib import Path

from src_to_implement.Optimization import Pooling

ID = 3  # identifier for dispatcher

class TestFullyConnected3(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1  # one-hot encoded labels
    
    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros(shape)
            weights[0] = 1
            weights[1] = 2
            return weights

    def test_trainable(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue(layer.trainable, msg="Possible error: The  trainable flag is not set to True. Please make sure"
                                             " to set set trainable=True.")

    def test_weights_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue((layer.weights.shape) in ((self.input_size + 1, self.output_size), (self.output_size, self.input_size + 1)))

    def test_forward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1],
                         self.output_size,
                         msg="Possible error: The shape of the output tensor is not correct. Make sure you combined the "
                             "weight matrix and the bias properly and extended the input such that the computation "
                             "includes weight multiplication and bias addition.")
        self.assertEqual(output_tensor.shape[0],
                         self.batch_size,
                         msg="Possible error: The batch size of the output tensor is not equal to the batch size of the"
                             " input tensor."
                         )
    
    def test_backward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        # print(output_tensor.shape)
        error_tensor = layer.backward(output_tensor)
        self.assertEqual(error_tensor.shape[1],
                         self.input_size,
                         msg="Possible error: The shape of the output tensor (backward function) is not correct. "
                             "Please make sure that you remove the fake errors that were comptet because of the added"
                             " collumn in the input (to include bias addition in the weight matrix multiplication)."
                         )
        self.assertEqual(error_tensor.shape[0],
                         self.batch_size,
                         msg="Possible error: The batch size of the output tensor is not equal to the batch size of the"
                             " input tensor (in the backward pass). Please refer to the exercise slides to use the "
                             "correct computation of the output tensor using the weights and the input."
                         )
    def test_update(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(self.input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            # print(error_tensor.shape)
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(self.input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)),
                            msg="Possible error: the weight update has not been performed correctly. Have a look at the"
                                "computation of gradient_weights. If the gradient_weights test passes, have a look at "
                                "the optimizer. Please also make sure that the weights are updated in the backward pass"
                                "if an optimizer is set!")

    def test_update_bias(self):
        input_tensor = np.zeros([self.batch_size, self.input_size])
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        layer.optimizer = Optimizers.Sgd(1)
        for _ in range(10):
            output_tensor = layer.forward(input_tensor)
            error_tensor = np.zeros([self.batch_size, self.output_size])
            error_tensor -= output_tensor
            layer.backward(error_tensor)
            new_output_tensor = layer.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)),
                            msg="Possible error: the update of the bias has not been performed correctly. Have a look at the"
                                "computation of gradient_weights. If the gradient_weights test passes, have a look at "
                                "the optimizer. Please also make sure that the weights are updated in the backward pass"
                                "if an optimizer is set!"
                            )

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers3.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference),
                             1e-5,
                             msg="Possible error: The gradient with respect to the input is not correct. Please refer "
                                 "to the exercise slides to use the correct computation of the output tensor using "
                                 "the weights and the input.")

    def test_gradient_weights(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        layers = list()
        layers.append(FullyConnected.FullyConnected(self.input_size, self.categories))
        layers.append(L2Loss())
        difference = Helpers3.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference),
                             1e-5,
                             msg="Possible error: The gradient with respect to the weights is not correct. Please refer "
                                 "to the exercise slides to use the correct computation of the output tensor using "
                                 "the weights and the input. Please also make sure that you implemented the "
                                 "gradients_weights property and store the gradient weights in this variable."
                             )

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected.FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0,
                           msg="Possible error: The initialization of the bias (i.e. the weights if stored in "
                               "single matrix) may be wrong. Make sure bias and weights are initialized randomly"
                               " between 0 and 1!")
        
    def test_initialization(self):
        input_size = 4
        categories = 10
        layer = FullyConnected.FullyConnected(input_size, categories)
        init = TestFullyConnected3.TestInitializer()
        layer.initialize(init, Initializers.Constant(0.5))
        self.assertEqual(init.fan_in, input_size, "Possible reason: Shape of the weights is not correct, check their"
                                                  "initialization.")
        self.assertEqual(init.fan_out, categories, "Possible reason: Shape of the weights is not correct, check their"
                                                   "initialization.")
        if layer.weights.shape[0]>layer.weights.shape[1]:
            self.assertLessEqual(np.sum(layer.weights) - 17, 1e-5, "Possible reason: Weights and/or bias is not"
                                                                   "initialized correctly, check their values.")
        else:
            self.assertLessEqual(np.sum(layer.weights) - 35, 1e-5, "Possible reason: Weights and/or bias is not"
                                                                   "initialized correctly, check their values.")

class TestReLU2(unittest.TestCase):
    def setUp(self):
        self.input_size = 5
        self.batch_size = 10
        self.half_batch_size = int(self.batch_size / 2)
        self.input_tensor = np.ones([self.batch_size, self.input_size])
        self.input_tensor[0:self.half_batch_size, :] -= 2

        self.label_tensor = np.zeros([self.batch_size, self.input_size])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.input_size)] = 1

    def test_trainable(self):
        layer = ReLU.ReLU()
        self.assertFalse(layer.trainable,
                         msg="Possible error: Trainable flag is set to true. Make sure it is set to False.")

    def test_forward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 1

        layer = ReLU.ReLU()
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0,
                         msg="Possible error: the ReLU function is not properly implemented. Make sure that the function"
                             "sets all negative values to zero and passes all positive values to the next layer"
                             " as they are according to ReLU(x) = max(0, x). ")

    def test_backward(self):
        expected_tensor = np.zeros([self.batch_size, self.input_size])
        expected_tensor[self.half_batch_size:self.batch_size, :] = 2

        layer = ReLU.ReLU()
        layer.forward(self.input_tensor)
        output_tensor = layer.backward(self.input_tensor * 2)
        self.assertEqual(np.sum(np.power(output_tensor - expected_tensor, 2)), 0,
                         msg="Possible error: The derivative of the ReLU function is not correctly implemented. Please"
                             " refer to the lecture slides for the correct derivative. Hint: you may need the input "
                             "tensor X of the forward pass also in the backward pass...")

    def test_gradient(self):
        input_tensor = np.abs(np.random.random((self.batch_size, self.input_size)))
        input_tensor *= 2.
        input_tensor -= 1.
        layers = list()
        layers.append(ReLU.ReLU())
        layers.append(L2Loss())
        difference = Helpers3.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5,
                             msg="Possible error: The derivative of the ReLU function is not correctly implemented. Please"
                             " refer to the lecture slides for the correct derivative. Hint: you may need the input "
                             "tensor X of the forward pass also in the backward pass...")

class TestSoftMax2(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = SoftMax.SoftMax()
        self.assertFalse(layer.trainable,
                         msg="Possible error: Trainable flag is set to true. Make sure it is set to False.")

    def test_forward_shift(self):
        input_tensor = np.zeros([self.batch_size, self.categories]) + 10000.
        layer = SoftMax.SoftMax()
        pred = layer.forward(input_tensor)
        self.assertFalse(np.isnan(np.sum(pred)),
                         msg="Possible error: The input tensor is not shifted to the negative domain. Please make sure "
                             "shift the input linearly to the negative domain in order to keep numerical stability.")

    def test_forward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertLess(loss, 1e-10,
                        msg="Possible error: The forward function is not implemented correctly. Please refer to the "
                            "lecture slides for help. Hint: The output of the SoftMax function corresponds to a "
                            "probability distribution to with the probabilities for each label (usually) in a"
                            " classification task. So check if the sum of the output is equal to 1!")

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12,
                               msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                   " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                   "Y from the forward pass also in the backward pass... "
                               )


    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.,
                           msg="Possible error: The forward function is not implemented correctly. Please refer to the "
                            "lecture slides for help. Hint: The output of the SoftMax function corresponds to a "
                            "probability distribution to with the probabilities for each label (usually) in a"
                            " classification task. So, check if the sum of the output is equal to 1!")

    def test_backward_zero_loss(self):
        input_tensor = self.label_tensor * 100.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()
        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        self.assertAlmostEqual(np.sum(error), 0,
                               msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                   " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                   "Y from the forward pass also in the backward pass... "
                                   "The test also fails if the CrossEntropyLoss() function is not yet or wrong implemented.")

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12,
                               msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                   "Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                   "Y from the forward pass also in the backward pass... "
                               )

    def test_regression_backward_high_loss_w_CrossEntropy(self):
        input_tensor = self.label_tensor - 1
        input_tensor *= -10.
        layer = SoftMax.SoftMax()
        loss_layer = Loss.CrossEntropyLoss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = loss_layer.backward(self.label_tensor)
        error = layer.backward(error)
        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertAlmostEqual(element, 1/3, places = 3,
                                   msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                       " The class confidence for wrong predicted lables is not decreased in the backward function."
                                       " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                       "Y from the forward pass also in the backward pass."
                                       "The test also fails if the CrossEntropyLoss() function is not yet or wrong implemented."
                                   )

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertAlmostEqual(element, -1, places = 3,
                                   msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                       " The class confidence for correct predicted lables is not increased in the backward function."
                                       " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                       "Y from the forward pass also in the backward pass."
                                       "The test also fails if the CrossEntropyLoss() function is not yet or wrong implemented."
                                   )


    def test_regression_forward(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)

        # just see if it's bigger then zero
        self.assertGreater(float(loss), 0.,
                           msg="Possible error: The forward function is not implemented correctly. Please refer to the "
                            "lecture slides for help. Hint: The output of the SoftMax function corresponds to a "
                            "probability distribution to with the probabilities for each label (usually) in a"
                            " classification task. So, check if the sum of the output is equal to 1!")


    def test_regression_backward(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()

        pred = layer.forward(input_tensor)
        loss_layer.forward(pred, self.label_tensor)
        error = layer.backward(self.label_tensor)

        # test if every wrong class confidence is decreased
        for element in error[self.label_tensor == 0]:
            self.assertLessEqual(element, 0,
                                 msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                     " The class confidence for wrong predicted lables is not decreased in the backward function."
                                     " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                     "Y from the forward pass also in the backward pass."
                                 )

        # test if every correct class confidence is increased
        for element in error[self.label_tensor == 1]:
            self.assertGreaterEqual(element, 0,
                                    msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                        " The class confidence for wrong predicted lables is not decreased in the backward function."
                                        " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                        "Y from the forward pass also in the backward pass."
                                    )

    def test_gradient(self):
        input_tensor = np.abs(np.random.random(self.label_tensor.shape))
        layers = list()
        layers.append(SoftMax.SoftMax())
        layers.append(L2Loss())
        difference = Helpers3.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-5,
                             msg="Possible error: The derivative of the ReLU function is not correctly implemented."
                                 " Please refer to the lecture slides for help. Hint: You may need the output tensor "
                                        "Y from the forward pass also in the backward pass. Also make sure to do the "
                                 "necessary summation in the backward pass over the right axis."
                             )

    def test_predict(self):
        input_tensor = np.arange(self.categories * self.batch_size)
        input_tensor = input_tensor / 100.
        input_tensor = input_tensor.reshape((self.categories, self.batch_size))
        # print(input_tensor)
        layer = SoftMax.SoftMax()
        prediction = layer.forward(input_tensor.T)
        # print(prediction)
        expected_values = np.array([[0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724, 0.21732724,
                                     0.21732724, 0.21732724],
                                    [0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387, 0.23779387,
                                     0.23779387, 0.23779387],
                                    [0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794, 0.26018794,
                                     0.26018794, 0.26018794],
                                    [0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095, 0.28469095,
                                     0.28469095, 0.28469095]])
        # print(expected_values)
        # print(prediction)
        np.testing.assert_almost_equal(expected_values, prediction.T,
                                       err_msg="Possible error: The forward function is not properly implemented. "
                                               "Please refer to the lecture slided for the correct function. make "
                                               "sure to do the necessary summation in the forward pass over the right "
                                               "axis")


class TestCrossEntropyLoss(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_zero_loss(self):
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(self.label_tensor, self.label_tensor)
        self.assertAlmostEqual(loss, 0,
                               msg="Possible error: The forward function is not correctly implemented. Please refer to "
                                   "the lecture slides for the correct function. Hint: the label has to be multiplied"
                                   " with the negative log of the prediction -ylog(y').")

    def test_high_loss(self):
        label_tensor = np.zeros((self.batch_size, self.categories))
        label_tensor[:, 2] = 1
        input_tensor = np.zeros_like(label_tensor)
        input_tensor[:, 1] = 1
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(input_tensor, label_tensor)
        self.assertAlmostEqual(loss, 324.3928805, places = 4,
                               msg="Possible error: The forward function is not correctly implemented. Please refer to "
                                   "the lecture slides for the correct function. Hint: the label has to be multiplied"
                                   " with the negative log of the prediction -ylog(y')."
                               )


class TestNeuralNetwork3(unittest.TestCase):
    plot = True
    directory = './plots/'
    Path(directory).mkdir(parents=True, exist_ok=True)
    log = 'log.txt'


    def test_append_layer(self):
        # this test checks if your network actually appends layers and whether it copies the optimizer to these layers
        net = NeuralNetwork.NeuralNetwork(learning_rate=1)
        net.optimizer_class = Optimizers.Sgd
        fcl_1 = FullyConnected.FullyConnected(1, 1)
        net.append_layer(fcl_1)
        fcl_2 = FullyConnected.FullyConnected(1, 1)
        net.append_layer(fcl_2)

        self.assertEqual(len(net.layers), 2,
                         msg="Possible error: The append_layer function is not yet implemented or wrong implemented."
                             "Make sure that the NeuralNetwork class is able to add a layer and stores it in a list "
                             "called layers.")
        self.assertFalse(net.layers[0].optimizer is net.layers[1].optimizer,
                         msg="Possible error: The optimizer is not copied for each layer. Make sure to perform a "
                             "deepcopy of the optimizer and assign it to every trainable layer.")

    def test_data_access(self):
        net = NeuralNetwork.NeuralNetwork(learning_rate=1)
        net.optimizer_class = Optimizers.Sgd
        categories = 3
        input_size = 4
        net.data_layer = Helpers3.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()
        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        out = net.forward()
        out2 = net.forward()

        self.assertNotEqual(out, out2,
                            msg="Possible error: The Neural Network hat no access to the provided data. Make sure to "
                                "create an attribute data_layer in the constructor. Additionally, make sure that the "
                                "forward function calls the next() function of this attribute to get the next batch as"
                                " inout_tensor for the forward.")

    def test_iris_data(self):
        net = NeuralNetwork.NeuralNetwork(1e-3)
        net.optimizer_class = Optimizers.Sgd
        categories = 3
        input_size = 4
        net.data_layer = Helpers3.IrisData(50)
        net.loss_layer = Loss.CrossEntropyLoss()

        fcl_1 = FullyConnected.FullyConnected(input_size, categories)
        net.append_layer(fcl_1)
        net.append_layer(ReLU.ReLU())
        fcl_2 = FullyConnected.FullyConnected(categories, categories)
        net.append_layer(fcl_2)
        net.append_layer(SoftMax.SoftMax())

        net.train(4000)
        plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
        plt.plot(net.loss, '-x')
        plt.show()

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)
        index_maximum = np.argmax(results, axis=1)
        one_hot_vector = np.zeros_like(results)
        for i in range(one_hot_vector.shape[0]):
            one_hot_vector[i, index_maximum[i]] = 1

        correct = 0.
        wrong = 0.
        for column_results, column_labels in zip(one_hot_vector, labels):
            if column_results[column_labels > 0].all() > 0:
                correct += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)
        print('\nOn the Iris dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
        self.assertGreater(accuracy, 0.8,
                           msg="Your network is not learning. Make sure that the gradients are computed correctly"
                               " in all layers and make sure that the weights are updated in the backward functions. "
                               "Have a look at the displayed loss curve.")
        
    def test_digit_data(self):
        net = NeuralNetwork.NeuralNetwork(1e-4, Initializers.He(), Initializers.Constant(0.1))
        net.optimizer_class = Optimizers.Sgd

        input_image_shape = (1, 8, 8)
        conv_stride_shape = (1, 1)
        convolution_shape = (1, 3, 3)
        categories = 10
        batch_size = 200
        num_kernels = 4

        net.data_layer = Helpers3.DigitData(batch_size)
        net.loss_layer = Loss.CrossEntropyLoss()

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

        net.train(200)

        if TestNeuralNetwork3.plot:
            description = 'on_digit_data'
            fig = plt.figure('Loss function for training a Convnet on the Digit dataset')
            plt.plot(net.loss, '-x')
            fig.savefig(os.path.join(self.directory, "TestConvNet_" + description + ".pdf"), transparent=True,
                        bbox_inches='tight', pad_inches=0)

        data, labels = net.data_layer.get_test_set()

        results = net.test(data)

        accuracy = Helpers3.calculate_accuracy(results, labels)
        with open(self.log, 'a') as f:
            print('On the UCI ML hand-written digits dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%',
                  file=f)
        print('\nOn the UCI ML hand-written digits dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')
        self.assertGreater(accuracy, 0.5, "Possible reason: If unittests for the layers fail, fix those first. The"
                                          "training with your CNN doesn't work properly. Check whether the"
                                          "SoftConvTests give you only zeros in the gradient tensor differences. Also"
                                          "make sure, that your weights and bias get updated correctly.")
        

class TestOptimizers(unittest.TestCase):

    def test_sgd(self):
        optimizer = Optimizers.Sgd(1.)

        result = optimizer.calculate_update(1., 1.)
        np.testing.assert_almost_equal(result, np.array([0.]),
                                       err_msg="Possible error: The Sgd optimizer is not properly implemented. "
                                               "SGD is used by some other unittests. If these fail it could be caused "
                                               "by a wrong implementation of the SGD optimizer.")

        result = optimizer.calculate_update(result, 1.)
        np.testing.assert_almost_equal(result, np.array([-1.]),
                                       err_msg="Possible error: The Sgd optimizer is not properly implemented. "
                                               "SGD is used by some other unittests. If these fail it could be caused "
                                               "by a wrong implementation of the SGD optimizer."
                                       )

class L2Loss:
    def __init__(self):
        self.input_tensor = None

    def predict(self, input_tensor):
        return input_tensor

    def forward(self, input_tensor, label_tensor):
        self.input_tensor = input_tensor
        return np.sum(np.square(input_tensor - label_tensor))

    def backward(self, label_tensor):
        return 2*np.subtract(self.input_tensor, label_tensor)

class TestFlatten(unittest.TestCase):

    def setUp(self):
        self.batch_size = 9
        self.input_shape = (3, 4, 11)
        self.input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        self.input_tensor = self.input_tensor.reshape(self.batch_size, *self.input_shape)

    def test_trainable(self):
        layer = Flatten.Flatten()
        self.assertFalse(layer.trainable, "Possible reason: Flatten doesn't inherit from the base layer.")

    def test_flatten_forward(self):
        flatten = Flatten.Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        input_vector = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_vector = input_vector.reshape(self.batch_size, np.prod(self.input_shape))
        self.assertLessEqual(np.sum(np.abs(output_tensor-input_vector)), 1e-9,
                             "Possible reason: Input tensor is not reshaped correctly. Make sure to keep the batch"
                             "dimension and don't swap axes for the other dimensions, when you flatten them.")

    def test_flatten_backward(self):
        flatten = Flatten.Flatten()
        output_tensor = flatten.forward(self.input_tensor)
        backward_tensor = flatten.backward(output_tensor)
        self.assertLessEqual(np.sum(np.abs(self.input_tensor - backward_tensor)), 1e-9,
                             "Possible reason: Error tensor is not reshaped correctly, make sure to store the correct"
                             "shape in the forward pass.")

class TestDropout(unittest.TestCase):
    def setUp(self):
        self.batch_size = 10000
        self.input_size = 10
        self.input_tensor = np.ones((self.batch_size, self.input_size))

    def test_trainable(self):
        layer = Dropout.Dropout(0.25)
        self.assertFalse(layer.trainable, "Error: trainable flag is set to true.")

    def test_default_phase(self):
        drop_layer = Dropout.Dropout(0.25)
        self.assertFalse(drop_layer.testing_phase, "Error: default for testing_phase should be false.")

    def test_forward_trainTime(self):
        drop_layer = Dropout.Dropout(0.25)
        output = drop_layer.forward(self.input_tensor)
        self.assertEqual(np.max(output), 4)
        self.assertEqual(np.min(output), 0)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertAlmostEqual(sum_over_mean/self.input_size, 1., places=1,
                               msg="Make sure to implement inverted dropout:\n"
                                   "During training time:\n"
                                   "- randomly set activations to zero with probability '1-p'\n"
                                   "- multiply all activations by 1/p")

    def test_position_preservation(self):
        drop_layer = Dropout.Dropout(0.5)
        output = drop_layer.forward(self.input_tensor)
        error_prev = drop_layer.backward(self.input_tensor)
        np.testing.assert_almost_equal(np.where(output == 0.), np.where(error_prev == 0.),
                                       err_msg="Make sure to set the error tensor to zero in the"
                                               "positions where the input tensor was set to zero during"
                                               "the forward pass.")

    def test_forward_testTime(self):
        drop_layer = Dropout.Dropout(0.5)
        drop_layer.testing_phase = True
        output = drop_layer.forward(self.input_tensor)

        self.assertEqual(np.max(output), 1.)
        self.assertEqual(np.min(output), 1.)
        sum_over_mean = np.sum(np.mean(output, axis=0))
        self.assertEqual(sum_over_mean, 1. * self.input_size,
                         msg="Make sure that during test time the input tensor is left unchanged"
                             "by the 'forward' method.")

    def test_backward(self):
        drop_layer = Dropout.Dropout(0.5)
        drop_layer.forward(self.input_tensor)
        output = drop_layer.backward(self.input_tensor)
        self.assertEqual(np.max(output), 2, msg="Possible error: in the backward during training time"
                                            "all activations should be multiplied by 1/p")
        self.assertEqual(np.min(output), 0, msg="Possible error: in the backward the activations corresponding to those"
                                                "set to zero in the forward pass should also be set to zero.")

    def test_gradient(self):
        batch_size = 10
        input_size = 10
        input_tensor = np.ones((batch_size, input_size))
        label_tensor = np.zeros([batch_size, input_size])
        for i in range(batch_size):
            label_tensor[i, np.random.randint(0, input_size)] = 1
        layers = list()
        layers.append(Dropout.Dropout(0.5))
        layers.append(L2Loss())
        difference = Helpers3.gradient_check(layers, input_tensor, label_tensor, seed=1337)
        self.assertLessEqual(np.sum(difference), 1e-5)

class TestConv(unittest.TestCase):
    plot = False
    directory = 'plots/'

    class TestInitializer:
        def __init__(self):
            self.fan_in = None
            self.fan_out = None

        def initialize(self, shape, fan_in, fan_out):
            self.fan_in = fan_in
            self.fan_out = fan_out
            weights = np.zeros((1, 3, 3, 3))
            weights[0, 1, 1, 1] = 1
            return weights

    def setUp(self):
        self.batch_size = 2
        self.input_shape = (3, 10, 14)
        self.input_size = 14 * 10 * 3
        self.uneven_input_shape = (3, 11, 15)
        self.uneven_input_size = 15 * 11 * 3
        self.spatial_input_shape = np.prod(self.input_shape[1:])
        self.kernel_shape = (3, 5, 8)
        self.num_kernels = 4
        self.hidden_channels = 3

        self.categories = 105
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1

    def test_trainable(self):
        layer = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        self.assertTrue(layer.trainable, "Possible reason: Convolutional Layer doesn't inherit the Base Layer and/or"
                                         "attribute trainable isn't set in the Convolutional Layer.")

    def test_forward_size(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]),
                         "Possible reason: Shape of the output_tensor is not correct. Make sure to have a same"
                         "correlation in the image dimensions and a valid correlation in the channel dimension. The"
                         "desired output shape is Batches x #kernels x spatialX x (spatialY).")

    def test_forward_size_stride(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(int(np.prod(self.input_shape) * self.batch_size)), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, 4, 7),
                         "Possible reason: If test_forward_size fails as well, fix that one first. Then, if this still"
                         "fails, your subsampling doesn't work properly. It might help, to do the subsampling on"
                         "paper and compare it to the intermediate results of your code.")

    def test_forward_size_stride_uneven_image(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels + 1)
        input_tensor = np.array(range(int(np.prod(self.uneven_input_shape) * (self.batch_size + 1))), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size + 1, *self.uneven_input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, ( self.batch_size+1, self.num_kernels+1, 4, 8),
                         "Possible reason: If test_forward_size and/or test_forward_size_stride fail as well, fix those"
                         "first. If they don't, then your subsampling doesn't work for an uneven sized image. Think"
                         "about what happens to the last few pixels, when doing this subsampling.")

    def test_forward(self):
        np.random.seed(1337)
        conv = Conv.Conv((1, 1), (1, 3, 3), 1)
        conv.weights = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        input_tensor = np.random.random((1, 1, 10, 14))
        expected_output = gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertAlmostEqual(difference, 0., places=1,
                               msg="Possible reason: Implementation for the forward pass is not correct. Check the"
                                   "correlation again and make sure, that you correlate the input_tensor with the"
                                   "correct kernel.")

    def test_forward_multi_channel(self):
        np.random.seed(1337)
        maps_in = 2
        bias = 1
        conv = Conv.Conv((1, 1), (maps_in, 3, 3), 1)
        filter = (1./15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.weights = np.repeat(filter[None, ...], maps_in, axis=1)
        conv.bias = np.array([bias])
        input_tensor = np.random.random((1, maps_in, 10, 14))
        expected_output = bias
        for map_i in range(maps_in):
            expected_output = expected_output + gaussian_filter(input_tensor[0, map_i, :, :], 0.85, mode='constant',
                                                                cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor) / maps_in)
        self.assertAlmostEqual(difference, 0., places=1,
                               msg="Possible reason: If test_forward fails as well, fix that one first. Otherwise,"
                                   "your implementation for the valid correlation in channel dimension might not work"
                                   "correctly. Make sure to select the correct channel if you do a same correlation."
                                   "Also make sure to add the bias.")

    def test_forward_fully_connected_channels(self):
        np.random.seed(1337)
        conv = Conv.Conv((1, 1), (3, 3, 3), 1)
        conv.weights = (1. / 15.) * np.array([[[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]], [[1, 2, 1], [2, 3, 2], [1, 2, 1]]])
        conv.bias = np.array([0])
        conv.weights = np.expand_dims(conv.weights, 0)
        tensor = np.random.random((1, 1, 10, 14))
        input_tensor = np.zeros((1, 3, 10, 14))
        input_tensor[:, 0] = tensor.copy()
        input_tensor[:, 1] = tensor.copy()
        input_tensor[:, 2] = tensor.copy()
        expected_output = 3 * gaussian_filter(input_tensor[0, 0, :, :], 0.85, mode='constant', cval=0.0, truncate=1.0)
        output_tensor = conv.forward(input_tensor).reshape((10, 14))
        difference = np.max(np.abs(expected_output - output_tensor))
        self.assertLess(difference, 0.2, "Possible reason: If test_forward and/or test_forward_multi_channel fail as"
                                         "well, fix those first. Otherwise, your implementation for the valid"
                                         "correlation in channel dimension might not work correctly. Make sure to"
                                         "select the correct channel if you do a same correlation. It might help, to"
                                         "do the correlation on paper to get a better understanding of it.")


    def test_backward_size(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape),
                         "Possible reason: Shape of the error_tensor is not correct. Make sure to have a same"
                         "correlation in the image dimensions and a valid correlation in the channel dimension. The"
                         "desired output shape is batches x channels x spatialX x (spatialY). Also make sure to"
                         "reslice the kernels correctly and flip them, if necessary. It might help, to do the reslicing"
                         "on paper and compare your results to the ones from your code.")

    def test_backward_size_stride(self):
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape),
                         "Possible reason: If test_backward_size fails as well, fix that one first. Otherwise, the"
                         "upsampling of the error_tensor might not work properly. It might help, to do the upsampling"
                         "on paper and compare your results.")

    def test_1x1_convolution(self):
        conv = Conv.Conv((1, 1), (3, 1, 1), self.num_kernels)
        input_tensor = np.array(range(self.input_size * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertEqual(output_tensor.shape, (self.batch_size, self.num_kernels, *self.input_shape[1:]))
        error_tensor = conv.backward(output_tensor)
        self.assertEqual(error_tensor.shape, (self.batch_size, *self.input_shape),
                         "Possible reason: 1x1 convolution doesn't work for the backward pass. If any tests for the"
                         "backward pass fail, fix those first. Otherwise, make sure, that the kernel size of 1x1 does"
                         "not cause any unexpected behaviour in your code.")

    def test_layout_preservation(self):
        conv = Conv.Conv((1, 1), (3, 3, 3), 1)
        # conv.initialize(self.TestInitializer(), Initializers.Constant(0.0))
        filter_elements = 9
        fan_in = filter_elements * 3
        fan_out = filter_elements * 1
        conv.bias = np.zeros_like(conv.bias)
        conv.weights = self.TestInitializer().initialize(conv.weights.shape, fan_in, fan_out)
        input_tensor = np.array(range(np.prod(self.input_shape) * self.batch_size), dtype=float)
        input_tensor = input_tensor.reshape(self.batch_size, *self.input_shape)
        output_tensor = conv.forward(input_tensor)
        self.assertAlmostEqual(np.sum(np.abs(np.squeeze(output_tensor) - input_tensor[:, 1, :, :])), 0.,
                               msg="Possible reason: If any other tests for the forward pass fail, fix those first."
                                   "Otherwise, in the forward pass, make sure that you store the results of the"
                                   "correlation in the correct places for the output_tensor.")

    def test_gradient(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers3.gradient_check(layers, input_tensor, self.label_tensor)
        self.assertLessEqual(np.sum(difference), 5e-2,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass fail, fix"
                             "those first. Otherwise, your gradient w.r.t. lower layers is not correct. Check the"
                             "SoftConvTests for more detailed information as well. Make sure, that you have implemented"
                             "the valid convolution across the channel dimension correctly. Also check, whether your"
                             "kernels are resliced and flipped correctly. If you pad the error_tensor, make sure that"
                             "the padding is correct.")

    def test_gradient_weights(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers3.gradient_check_weights(layers, input_tensor, self.label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass fail, fix"
                             "those first. Otherwise your gradient w.r.t weights is not correct. Check the"
                             "SoftConvTests for more detailed information as well. Make sure, that you have implemented"
                             "the valid convolution across the channel dimension correctly. If you pad the input_tensor"
                             "make sure, that the padding is correct.")

    def test_gradient_weights_strided(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 36])
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((2, 2), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers3.gradient_check_weights(layers, input_tensor, label_tensor, False)
        self.assertLessEqual(np.sum(difference), 1e-5,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass and/or"
                             "test_gradient_weights fail, fix those first. Otherwise your gradient w.r.t weights is not"
                             "correct in the case of stride. Check the SoftConvTests for more detailed information as"
                             "well. Check, if the upsampling of the error_tensor works correctly.")

    def test_gradient_bias(self):
        np.random.seed(1337)
        input_tensor = np.abs(np.random.random((2, 3, 5, 7)))
        layers = list()
        layers.append(Conv.Conv((1, 1), (3, 3, 3), self.hidden_channels))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers3.gradient_check_weights(layers, input_tensor, self.label_tensor, True)

        self.assertLessEqual(np.sum(difference), 1e-5,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass fail, fix"
                             "those first. Otherwise, your gradient w.r.t bias is not correct. Check the SoftConvTests"
                             "for more detailed information as well. Make sure, that you sum up the error_tensor over"
                             "the correct dimensions, as described in the formula on slide 18 of the exercise slides.")

    def test_weights_init(self):
        # simply checks whether you have not initialized everything with zeros
        conv = Conv.Conv((1, 1), (100, 10, 10), 150)
        self.assertGreater(np.mean(np.abs(conv.weights)), 1e-3,
                           "Possible reason: Don't initialize your weights with zeros, use a random initialization"
                           "instead.")

    def test_bias_init(self):
        conv = Conv.Conv((1, 1), (1, 1, 1), 150 * 100 * 10 * 10)
        self.assertGreater(np.mean(np.abs(conv.bias)), 1e-3,
                           "Possible reason: Don't initialize your bias with zeros, use a random initialization"
                           "instead.")

    def test_gradient_stride(self):
        np.random.seed(1337)
        label_tensor = np.random.random([self.batch_size, 35])
        input_tensor = np.abs(np.random.random((2, 6, 5, 14)))
        layers = list()
        layers.append(Conv.Conv((1, 2), (6, 3, 3), 1))
        layers.append(Flatten.Flatten())
        layers.append(L2Loss())
        difference = Helpers3.gradient_check(layers, input_tensor, label_tensor)
        self.assertLessEqual(np.sum(difference), 1e-4,
                             "Possible reason: If any tests for your Flatten Layer and/or the forward pass and/or the"
                             "test_gradient fail, fix those first. Otherwise, your gradient w.r.t. lower layers is not"
                             "correct in case of stride. Check the SoftConvTests for more detailed information as well."
                             "Make sure, that the upsampling of the error_tensor works correctly. Also check, if the"
                             "valid convolution across the channel dimension is working correctly for the forward and"
                             "backward pass.")

    def test_update(self):
        input_tensor = np.random.uniform(-1, 1, (self.batch_size, *self.input_shape))
        conv = Conv.Conv((3, 2), self.kernel_shape, self.num_kernels)
        conv.optimizer = Optimizers.Sgd(1)
        #conv.initialize(Initializers.He(), Initializers.Constant(0.1))
        # conv.weights = np.random.rand(4, 3, 5, 8)
        # conv.bias = 0.1 * np.ones(4)
        for _ in range(10):
            output_tensor = conv.forward(input_tensor)
            error_tensor = np.zeros_like(output_tensor)
            error_tensor -= output_tensor
            conv.backward(error_tensor)
            new_output_tensor = conv.forward(input_tensor)
            self.assertLess(np.sum(np.power(output_tensor, 2)), np.sum(np.power(new_output_tensor, 2)),
                            "Possible reason: If any other tests fail, fix those first. Make sure, that the weights and"
                            "the bias get updated with the respective optimizer. Also check the SoftConvTests to see,"
                            "whether your gradients are correct.")

    def test_initialization(self):
        conv = Conv.Conv((1, 1), self.kernel_shape, self.num_kernels)
        init = TestConv.TestInitializer()
        conv.initialize(init, Initializers.Constant(0.1))
        self.assertEqual(init.fan_in, np.prod(self.kernel_shape),
                         "Possible reason: Calculation of fan_in in your initialize method is not correct.")
        self.assertEqual(init.fan_out, np.prod(self.kernel_shape[1:]) * self.num_kernels,
                         "Possible reason: Calculation of fan_out in your initialize method is not correct.")


if __name__ == '__main__':

    unittest.main()
