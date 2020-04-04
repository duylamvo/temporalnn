"""Time series using Convolution Neural Network."""

import math

from keras.models import Sequential, Model
from keras.layers import (Input, Dense, Flatten, LSTM, Activation, Multiply,
                          TimeDistributed, RepeatVector, BatchNormalization)
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import backend, regularizers, layers as kl
from keras.utils.vis_utils import plot_model

from PIL import Image


class TemporalNN(object):
    """Temporal Neural Network class in general."""

    @staticmethod
    def build_model(**kwargs):
        """Build model."""
        pass

    @staticmethod
    def validate_model(**kwargs):
        """Validate the model."""
        pass

    @staticmethod
    def train_model(model, inputs, outputs, epochs=1000, verbose=1):
        """Train model."""
        model.fit(inputs, outputs, epochs=epochs, verbose=verbose)

    @staticmethod
    def predict(**kwargs):
        """Predict model with matching input."""
        pass

    @staticmethod
    def debug_layer(test_input, tensor_in, tensor_out):
        """Debug from input layers to another layer with given data test.

        Example:
            X[1] ~ [[1], [2], [3], [4], [5]] ~ (4, 1) -> need to have [[X[1]]] to match input shape
            print(debug_layer([[X[1]]], [input_layer], [dilated_causal]))

        """
        debug_func = backend.function(tensor_in, tensor_out)
        debug_out = debug_func(test_input)
        return debug_out

    @staticmethod
    def describe(model, img_file=None):
        """Describe model.

        :param model:
        :param img_file: where to save the picture
        :return:
        """
        model.summary()
        if img_file:
            plot_model(model, to_file=img_file,
                       show_layer_names=True, show_shapes=True)
            Image.open(img_file).show()

    @staticmethod
    def plot_model(model, to_file):
        """Save architecture of model to image file."""
        plot_model(model, to_file=to_file,
                   show_layer_names=True, show_shapes=True)


class SimpleTCNN(TemporalNN):
    """Simple Temporal Neural Network using CNN."""

    def __init__(self, input_shape=None, model=None):
        """Create instance of simple temporal convolutional neural network."""
        self.input_shape = input_shape
        self.model = model

    @staticmethod
    def build_model(input_shape):
        """Build model.

        Ref: https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/
        :param input_shape: shape of the input
        :return: keras model
        """
        model = Sequential()
        model.add(Conv1D(filters=64,
                         kernel_size=3,
                         activation='relu',
                         padding='same',
                         input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=["acc"])

        return model


class SimpleLSTM(TemporalNN):
    """Simple Temporal Neural Network using LSTM."""

    def __init__(self, input_shape=None, model=None):
        """Create instance of simple temporal LSTM neural network."""
        self.input_shape = input_shape
        self.model = model

    @staticmethod
    def build_model(input_shape):
        """Build model.

        Ref: https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/
        :param input_shape: shape of the input
        :return: keras model
        """
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=input_shape))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        return model


class SimpleCNNLSTM(TemporalNN):
    """Simple Temporal Neural Network using CNN -->LSTM.

    This support very long input sequences -> can read as blocks or sequence by CNN models -> pierce into LSTM model
    An n-steps time series as input could be divided into many subsequences of 2 time_steps. each of 2 time_steps will
    be fed into n/2 features and then continued being flatten and fed into LSTM network

    This helps the model can handle a large number of input at the same time. We need at fist time, divide a n ts to
    n/2 sample, each sample has 2 time_steps for example.
    The model here using TimeDistributed wrapper to apply for each subsequence in the sample
    """

    def __init__(self, input_shape=None, model=None):
        """Create instance of simple temporal LSTM neural network."""
        self.input_shape = input_shape
        self.model = model

    @staticmethod
    def build_model(input_shape):
        """Build model.

        Ref: https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/
        :param input_shape: shape of the input. how many step per sub-sequence
        :return: keras model
        """
        model = Sequential()
        model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1,
                                         activation='relu'), input_shape=input_shape))
        model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
        model.add(TimeDistributed(Flatten()))
        model.add(LSTM(units=50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        return model


class SimpleEncoderDecoder(TemporalNN):
    """Simple Temporal Neural Network using 2 block of LSTM as Encoder-Decoder to predict next 2 values as output.

    An LSTM encoder to read and encode the input sequences of n time steps.
    The encoded sequence will be repeated 2 times by the model for the two output time steps required by the model
    using a RepeatVector layer.

    These will be fed to a decoder LSTM layer -> a Dense output layer wrapped in a TimeDistributed layer that will
    produce one output for each step in the output sequence.

    estimate one time more steps (multi-steps) forward
    """

    def __init__(self, input_shape=None, model=None):
        """Create instance of simple temporal LSTM neural network."""
        self.input_shape = input_shape
        self.model = model

    @staticmethod
    def build_model(input_shape):
        """Build model.

        Ref: https://machinelearningmastery.com/how-to-get-started-with-deep-learning-for-time-series-forecasting-7-day-mini-course/
        :param input_shape: shape of the input. how many step per sub-sequence
        :return: keras model
        """
        model = Sequential()
        model.add(LSTM(units=100, activation='relu', input_shape=input_shape))
        model.add(RepeatVector(2))
        model.add(LSTM(units=100, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')

        return model


class WaveNet(TemporalNN):
    """WaveNet model using dilated causal neural network."""

    def __init__(self, input_shape=None, model=None, kernel_size=2):
        """Create Wavenet model for forecasting."""
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.model = model

    @staticmethod
    def build_wavenet_layer(inputs,
                            dilation_rate,
                            num_filters=32,
                            kernel_size=2,
                            gated_activations=None,
                            batch_normalization=False,
                            name="wavenet"):
        """Wave-Net blocks which including dilated causal layers.

        Wavenet Paper: arXiv:1609.03499v2, https://arxiv.org/pdf/1609.03499.pdf

        A WaveNet was built on top of causal layers with windows size (kernel) = 2. The dilated rate will changed in order
        of the kernel size. The paper mentioned to use slice of 2 hence in this implementation we also use 2. In practice,
        it is not the matter to change kernel_size more than 2.

        The dilated rate will change for each layers in order of 2, for instance, ex: 1, 2, 2², 2³, ..
        If the number of dilated_layers is not defined, then the log(input, base 2) will be used.

        The activation of a wavenet block is done through gated unit activation which is tanh(x) * sigmoid(x). Here the
        multiplication is element-wise. There are also different improved version for this gated unit activation to
        improve convergence, ex, ReLu(x) * sigmoid(x).

        The final output of WavNet is a 1x1 convolution, output to another layers or a ReLu activation.

        """
        # input_layer = kl.Input(shape=X.shape[1:], name='main_input')
        x = inputs
        dilated_causal = Conv1D(num_filters,
                                kernel_size=kernel_size,
                                padding="causal",
                                dilation_rate=dilation_rate,
                                # kernel_initializer='zeros',                # ones or zeros to debug
                                # https://keras.io/examples/cifar10_resnet/
                                kernel_regularizer=regularizers.l2(1e-4),
                                name="{}_dilated_causal".format(name)
                                )
        x = dilated_causal(x)

        if batch_normalization:
            x = BatchNormalization(name="{}_BN".format(name))(x)

        # Gated Function, according to paper it is
        # gated unit = tanh(x) * sigmoid(x) -> if want users also can use relu(x) * sigmoid(x)
        gated_activations = gated_activations or ['tanh', 'sigmoid']
        if isinstance(gated_activations, list):
            acts = []
            for act in gated_activations:
                assert isinstance(
                    act, str), "activations accept only string, ex: relu, tanh, sigmoid, etc."
                acts = acts + [Activation(act, name="{}_{}_x".format(name, act))(x)]
            if len(acts) > 0:
                x = Multiply(name="{}_gated_unit_activation".format(name))(acts)

        x = Conv1D(num_filters, kernel_size=1, name="{}_1x1".format(name))(x)

        return x

    @staticmethod
    def build_model(input_shape,
                    n_in_steps=5,
                    n_out_steps=1,
                    n_dilated_layers=None,
                    gated_activations=None,
                    kernel_size=2,
                    n_conv_filters=32):
        """Build WaveNet Model.

        WaveNet paper haves kernel size (or windows size) = 2, but it is up to you to implement it.
        Hence the dilation rate will be an order of base kernel size.

        For example with kernel_size = 2 (default), then the dilation rates will be as follows:
        1, 2, 2², 2³, this depends on how many layers you defined.

        Number of dilated layers also not so big. If you have 1024 input steps, the number of layers
        should be 10, that is 2 power 10. It does not make sense, if larger than 10 for 1024 input
        steps, because after layer 10th, only 1 output is remaining active.

        Default wavenet will have kernel_size = 2

        """
        n_dilated_layers = n_dilated_layers or int(
            math.log(n_in_steps, kernel_size))

        main_inputs = Input(shape=input_shape, name='main_input')
        x = main_inputs
        for i in range(n_dilated_layers):
            y = WaveNet.build_wavenet_layer(x,
                                            num_filters=n_conv_filters,
                                            dilation_rate=kernel_size ** i,
                                            gated_activations=gated_activations,
                                            batch_normalization=True,
                                            name="wavenet-{}".format(i))

            # if first loop, then do not add residual net (skip connection)
            if i == 0:
                x = y
            else:
                x = kl.add([x, y])
        x = Activation('relu', name='ReLu_1')(x)
        x = Conv1D(n_conv_filters, kernel_size=1,
                   padding='same', activation='relu')(x)
        x = Conv1D(n_conv_filters, kernel_size=1,
                   padding='same', activation='relu')(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)

        # TODO: How to support multi-steps multi-columns outputs ?
        #  for-loop in creating models at final
        final = Dense(n_out_steps)(x)

        return Model(inputs=main_inputs, outputs=final)
