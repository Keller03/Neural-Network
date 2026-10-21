import unittest
from Layers import FullyConnected, ReLU, SoftMax
from Optimization import Loss
import numpy as np
import NeuralNetwork
import Helpers

ID = 1  # identifier for dispatcher

class TestFullyConnected(unittest.TestCase):
    def setUp(self):
        self.batch_size = 9
        self.input_size = 4
        self.output_size = 3
        self.input_tensor = np.random.rand(self.batch_size, self.input_size)

        self.categories = 4
        self.label_tensor = np.zeros([self.batch_size, self.categories])
        for i in range(self.batch_size):
            self.label_tensor[i, np.random.randint(0, self.categories)] = 1  # one-hot encoded labels

    def test_trainable(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue(layer.trainable, msg="Possible error: The trainable flag is not set to True. Please make sure"
                                             " to set trainable=True.")

    def test_weights_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        self.assertTrue((layer.weights.shape) in ((self.input_size + 1, self.output_size), (self.output_size, self.input_size + 1)))

    def test_forward_size(self):
        layer = FullyConnected.FullyConnected(self.input_size, self.output_size)
        output_tensor = layer.forward(self.input_tensor)
        self.assertEqual(output_tensor.shape[1],
                         self.output_size,
                         msg="Possible error: The shape of the output tensor is not correct. Make sure you combine the "
                             "weight matrix and the bias properly and extended the input such that the computation "
                             "includes weight multiplication and bias addition.")
        self.assertEqual(output_tensor.shape[0],
                         self.batch_size,
                         msg="Possible error: The batch size of the output tensor is not equal to the batch size of the"
                             " input tensor."
                         )

    def test_bias(self):
        input_tensor = np.zeros((1, 100000))
        layer = FullyConnected.FullyConnected(100000, 1)
        result = layer.forward(input_tensor)
        self.assertGreater(np.sum(result), 0,
                           msg="Possible error: The initialization of the bias (i.e. the weights if stored in "
                               "single matrix) may be wrong. Make sure bias and weights are initialized randomly"
                               " between 0 and 1!")


class TestReLU(unittest.TestCase):
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

class TestSoftMax(unittest.TestCase):

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
                         msg="Possible error: The input tensor is not suitably shifted to the negative domain. Please make sure "
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
                            "probability distribution with the probabilities for each label (usually) in a"
                            " classification task. So check if the sum of the output is equal to 1!")

    def test_regression_high_loss(self):
        input_tensor = self.label_tensor - 1.
        input_tensor *= -100.
        layer = SoftMax.SoftMax()
        loss_layer = L2Loss()
        pred = layer.forward(input_tensor)
        loss = loss_layer.forward(pred, self.label_tensor)
        self.assertAlmostEqual(float(loss), 12,
                               msg="The forward function of the SoftMax is not implemented correctly. Please check the lecture & exercise slides."
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
                            "lecture and exercise slides for help. Hint: The output of the SoftMax function corresponds to a "
                            "probability distribution with the probabilities for each label (usually) in a"
                            " classification task. So, check if the sum of the output is equal to 1!")


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
                                               "Please refer to the lecture slides for the correct function. Make "
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
                                   "the lecture slides for the correct function.")

    def test_high_loss(self):
        label_tensor = np.zeros((self.batch_size, self.categories))
        label_tensor[:, 2] = 1
        input_tensor = np.zeros_like(label_tensor)
        input_tensor[:, 1] = 1
        layer = Loss.CrossEntropyLoss()
        loss = layer.forward(input_tensor, label_tensor)
        self.assertAlmostEqual(loss, 324.3928805, places = 4,
                               msg="Possible error: The forward function is not correctly implemented. Please refer to "
                                   "the lecture slides for the correct function."
                               )


class TestNeuralNetwork(unittest.TestCase):

    def test_append_layer(self):
        # this test checks if your network actually appends layers
        net = NeuralNetwork.NeuralNetwork()
        fcl_1 = FullyConnected.FullyConnected(1, 1)
        net.append_layer(fcl_1)
        fcl_2 = FullyConnected.FullyConnected(1, 1)
        net.append_layer(fcl_2)

        self.assertEqual(len(net.layers), 2,
                         msg="Possible error: The append_layer function is not yet implemented or implemented incorrectly. "
                             "Make sure that the NeuralNetwork class is able to add a layer and stores it in a list "
                             "called layers.")

    def test_data_access(self):
        net = NeuralNetwork.NeuralNetwork()
        categories = 3
        input_size = 4
        net.data_layer = Helpers.IrisData(50)
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
                            msg="Possible error: The Neural Network has no access to the provided data. Make sure to "
                                "create an attribute data_layer in the constructor. Additionally, make sure that the "
                                "forward function calls the next() function of this attribute to get the next batch as"
                                " input_tensor for the forward.")


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


if __name__ == '__main__':

    unittest.main()
    exit(0)
