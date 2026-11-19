import numpy as np  


class Initializer:
    """
    Initializer for node weights.
    """

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights according to some scheme.
        """

        raise NotImplementedError()


class Constant(Initializer):

    def __init__(self, value=0.1):
        self.value = value

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights with the given value.
        """
        return np.ones(weights_shape) * self.value


class UniformRandom(Initializer):

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights uniformly with values between 0 and 1.
        """
        return np.random.random_sample(weights_shape)


class Xavier(Initializer):

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights according to the Xavier scheme,
        see http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf .
        """
        std_dev = np.sqrt(2. / (fan_in + fan_out))

        return np.random.normal(0, std_dev, weights_shape)


class He(Initializer):

    def initialize(self, weights_shape, fan_in, fan_out):
        """
        Initialize weights according to the He scheme,
        see https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf .
        """
        std_dev = np.sqrt(2. / fan_in)

        return np.random.normal(0, std_dev, weights_shape)
