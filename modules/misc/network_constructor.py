#!/usr/bin/env python

from enum import Enum
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input, BatchNormalization, Dropout, Add, Subtract, \
    Lambda, UpSampling2D, Reshape
from tensorflow.keras.activations import relu
from tensorflow.keras import backend as K

from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow.keras.models import clone_model
from tensorflow.keras.initializers import RandomUniform, Orthogonal, Constant
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from .noisy_dense import NoisyDense
from .resnet import create_res_net12
import numpy as np
import tensorflow as tf
import os
os.environ["PATH"] += os.pathsep + 'C:/Graphviz/bin/'

NormalDense = Dense


class NetworkArchitecture(Enum):
    NONE = "None"
    SMALL_DENSE = "SmallDense"
    DENSE = "Dense"
    LARGE_DENSE = "LargeDense"
    CNN = "CNN"
    SHALLOW_CNN = "ShallowCNN"
    DEEP_CNN = "DeepCNN"
    SINGLE_DENSE = "SingleDense"
    CNN_BATCHNORM = "CNNBatchnorm"
    DEEP_CNN_BATCHNORM = "DeepCNNBatchnorm"
    CNN_ICM = "ICMCNN"
    RESNET12 = "ResNet12"


"""
This file contains methods to construct neural networks in Tensorflow given the type of network, the number of units
or filters as well as the desired input and output shapes.
"""


def construct_network(network_parameters):
    # Distinguish Dense vs. NoisyDense Layer (Only for DQN)
    global Dense
    if network_parameters.get("NoisyNetworks"):
        Dense = NoisyDense
    else:
        Dense = NormalDense

    # Construct the network Input
    network_input = build_network_input(network_parameters.get('Input'), network_parameters)

    if network_parameters.get("InputResize"):
        network_branches = []
        for net_input in network_input:
            if len(net_input.shape) == 4:
                x = Resizing(*network_parameters["InputResize"])(net_input)
                network_branches.append(x)
            else:
                network_branches.append(net_input)
    else:
        network_branches = network_input

    # Depending on the number and types of the inputs either keep them separate or connect them right here.
    x = connect_branches(network_branches, network_parameters)

    # Construct the network body/bodies
    network_body = build_network_body(x, network_parameters)

    # Construct the network output.
    network_output = build_network_output(network_body, network_parameters)

    # Create the final model and plot it.
    model = Model(inputs=network_input, outputs=network_output, name=network_parameters.get("NetworkType"))
    model.summary()
    
    try:
        plot_model(model, "plots/"+network_parameters.get("NetworkType") + ".png", show_shapes=True)
    except ImportError:
        print("Could not create a model plot for {}.".format(network_parameters.get("NetworkType")))

    if network_parameters.get('TargetNetwork'):
        target_model = clone_model(model)
        return model, target_model
    return model


def build_network_input(network_input_shapes, network_parameters):
    network_input = []
    for input_shapes in network_input_shapes:
        x = Input(input_shapes)
        network_input.append(x)
    return network_input


def connect_branches(network_input, network_parameters):
    branch_shape_lengths = [len(x.shape) for x in network_input]
    input_types = None

    if len(branch_shape_lengths) > 1:
        if len(set(branch_shape_lengths)) == 1:
            if branch_shape_lengths[0] == 4:
                input_types = "images"
            elif branch_shape_lengths[0] == 2:
                input_types = "vectors"
        else:
            input_types = "mixed"
            if network_parameters.get("Vec2Img"):
                input_types = "forced_images"
    else:
        return network_input

    # Case only images just concatenate the input layers.
    if input_types == "images":
        return [Concatenate(axis=-1)(network_input)]

    # Case only vectors append Dense Layer to each Input and concatenate.
    if input_types == "vectors":
        network_branches = []
        for net_input in network_input:
            network_branches.append(net_input)
        return [Concatenate(axis=-1)(network_branches)]

    # Case mixed inputs append Dense Layer to Vector inputs and concatenate images and vectors respectively.
    if input_types == "mixed":
        # Vector Branches
        vector_branches = []
        for net_input in [x for x in network_input if len(x.shape) == 2]:
            vector_branches.append(net_input)
        if len(vector_branches) > 1:
            vector_branches = Concatenate(axis=-1)(vector_branches)
        else:
            vector_branches = vector_branches[0]
        # Image Branches
        image_branches = []
        for net_input in [x for x in network_input if len(x.shape) == 4]:
            image_branches.append(net_input)
        if len(image_branches) > 1:
            image_branches = Concatenate(axis=-1)(image_branches)
        else:
            image_branches = image_branches[0]
        return [image_branches, vector_branches]

    if input_types == "forced_images":
        # Vector Branches
        image_branches = []
        for net_input in [x for x in network_input if len(x.shape) == 2]:
            print((1, 1, net_input.shape[-1]))
            x = Reshape((1, 1, net_input.shape[-1]))(net_input)
            x = UpSampling2D(size=(84, 84))(x)
            image_branches.append(x)
        # Image Branches
        for net_input in [x for x in network_input if len(x.shape) == 4]:
            image_branches.append(net_input)
        if len(image_branches) > 1:
            return [Concatenate(axis=-1)(image_branches)]


def build_network_body(network_input, network_parameters):
    network_branches = []
    units, filters = 0, 0
    for net_input in network_input:
        if len(net_input.shape) == 2:
            net_architecture = network_parameters.get("VectorNetworkArchitecture")
            units = network_parameters.get("Units")
        else:
            net_architecture = network_parameters.get("VisualNetworkArchitecture")
            filters = network_parameters.get("Filters")
        network_branches.append(get_network_component(net_input, net_architecture, network_parameters, units=units, filters=filters))
    network_body = connect_branches(network_branches, network_parameters)
    if len(network_branches) > 1:
        network_body = [get_network_component(network_body[0], "SingleDense", network_parameters, units=network_parameters.get("Units")*5)]
    return network_body[0]


def build_network_output(net_in, network_parameters):
    network_output = []
    for net_out_shape, net_out_act in zip(network_parameters.get('Output'),
                                          network_parameters.get('OutputActivation')):
        if net_out_shape:
            if type(net_out_shape) == tuple:
                net_out_shape = net_out_shape[0]

            if network_parameters.get('DuelingNetworks'):
                y = Dense(network_parameters.get('Units'), activation="selu")(net_in)
                z = Dense(network_parameters.get('Units'), activation="selu")(net_in)

                y = Dense(1, name="value")(y)
                z = Dense(net_out_shape, activation=net_out_act, name="advantage")(z)
                z_mean = Lambda(lambda a: K.mean(a[:, 1:], axis=1, keepdims=True), name="mean")(z)
                net_out = Add()([y, z])
                net_out = Subtract(name="action")([net_out, z_mean])
            else:
                if network_parameters.get('KernelInitializer') == "RandomUniform":
                    net_out = Dense(net_out_shape, activation=net_out_act,
                                    kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                                    bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))(net_in)
                elif network_parameters.get('KernelInitializer') == "Orthogonal":
                    net_out = Dense(net_out_shape, activation=net_out_act,
                                    kernel_initializer=Orthogonal(0.5),
                                    bias_initializer=Constant(0.0))(net_in)
                else:
                    net_out = Dense(net_out_shape, activation=net_out_act)(net_in)
            network_output.append(net_out)
        else:
            network_output.append(net_in)
    if network_parameters.get('LogStdOutput'):
        log_std = LogStdLayer(net_out_shape)(net_in)
        network_output.append(log_std)
    return network_output


def connect_network_branches(network_branches, network_parameters):
    if len(network_branches) == 1:
        return network_branches[0]

    x = Concatenate(axis=-1)(network_branches)
    x = Dense(network_parameters.get('Filters')*5)(x)
    return x


def get_network_component(net_inp, net_architecture, network_parameters, units=32, filters=32):
    if net_architecture == NetworkArchitecture.NONE.value:
        x = net_inp
    elif net_architecture == NetworkArchitecture.SINGLE_DENSE.value:
        if network_parameters.get('KernelInitializer') == "Orthogonal":
            x = Dense(units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(net_inp)
        else:
            x = Dense(units, activation='selu')(net_inp)

    elif net_architecture == NetworkArchitecture.SMALL_DENSE.value:
        if network_parameters.get('KernelInitializer') == "Orthogonal":
            x = Dense(units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(net_inp)
            x = Dense(units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
        else:
            x = Dense(units, activation='selu')(net_inp)
            x = Dense(units, activation='selu')(x)

    elif net_architecture == NetworkArchitecture.DENSE.value:
        if network_parameters.get('KernelInitializer') == "Orthogonal":
            x = Dense(units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(net_inp)
            x = Dense(units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
            x = Dense(2*units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
        else:
            x = Dense(units, activation='selu')(net_inp)
            x = Dense(units, activation='selu')(x)
            x = Dense(2*units, activation='selu')(x)

    elif net_architecture == NetworkArchitecture.LARGE_DENSE.value:
        if network_parameters.get('KernelInitializer') == "Orthogonal":
            x = Dense(units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(net_inp)
            x = Dense(units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
            x = Dense(2*units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
            x = Dense(2*units, activation='selu',
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
        else:
            x = Dense(units, activation='selu')(net_inp)
            x = Dense(units, activation='selu')(x)
            x = Dense(2*units, activation='selu')(x)
            x = Dense(2*units, activation='selu')(x)

    elif net_architecture == NetworkArchitecture.CNN.value:
        if network_parameters.get('KernelInitializer') == "Orthogonal":
            x = Conv2D(filters, kernel_size=8, strides=4, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(net_inp)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = Conv2D(filters*2, kernel_size=3, strides=1, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = Flatten()(x)
            x = Dense(512, activation="selu",
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
        else:
            x = Conv2D(filters, kernel_size=8, strides=4, activation="selu")(net_inp)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
            x = Conv2D(filters*2, kernel_size=3, strides=1, activation="selu")(x)
            x = Flatten()(x)
            x = Dense(512, activation="selu")(x)

    elif net_architecture == NetworkArchitecture.CNN_ICM.value:
        x = Conv2D(filters, kernel_size=3, strides=2, padding="same", activation="selu")(net_inp)
        x = Conv2D(filters, kernel_size=3, strides=2, padding="same", activation="selu")(x)
        x = Conv2D(filters, kernel_size=3, strides=2, padding="same", activation="selu")(x)
        x = Conv2D(filters, kernel_size=3, strides=2, padding="same", activation="selu")(x)
        x = Flatten()(x)

    elif net_architecture == NetworkArchitecture.CNN_BATCHNORM.value:
        if network_parameters.get('KernelInitializer') == "Orthogonal":
            x = Conv2D(filters, kernel_size=8, strides=4, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(net_inp)
            x = BatchNormalization()(x)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters*2, kernel_size=3, strides=1, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = BatchNormalization()(x)
            x = Flatten()(x)
            x = Dense(512, activation="selu",
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
        else:
            x = Conv2D(filters, kernel_size=8, strides=4, activation="selu")(net_inp)
            x = BatchNormalization()(x)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters*2, kernel_size=3, strides=1, activation="selu")(x)
            x = BatchNormalization()(x)
            x = Flatten()(x)
            x = Dense(512, activation="selu")(x)

    elif net_architecture == NetworkArchitecture.RESNET12.value:
        x = create_res_net12(net_inp, filters)

    elif net_architecture == NetworkArchitecture.SHALLOW_CNN.value:
        pass

    elif net_architecture == NetworkArchitecture.DEEP_CNN.value:
        if network_parameters.get('KernelInitializer') == "Orthogonal":
            x = Conv2D(filters, kernel_size=8, strides=4, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(net_inp)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = Conv2D(filters*4, kernel_size=3, strides=1, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = Flatten()(x)
            x = Dense(512, activation="selu",
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
        else:
            x = Conv2D(filters, kernel_size=8, strides=4, activation="selu")(net_inp)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
            x = Conv2D(filters*4, kernel_size=3, strides=1, activation="selu")(x)
            x = Flatten()(x)
            x = Dense(512, activation="selu")(x)

    elif net_architecture == NetworkArchitecture.DEEP_CNN_BATCHNORM.value:
        if network_parameters.get('KernelInitializer') == "Orthogonal":
            x = Conv2D(filters, kernel_size=8, strides=4, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(net_inp)
            x = BatchNormalization()(x)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters*4, kernel_size=3, strides=1, activation="selu",
                       kernel_initializer=Orthogonal(np.sqrt(2)),
                       bias_initializer=Constant(0.0))(x)
            x = BatchNormalization()(x)
            x = Flatten()(x)
            x = Dense(512, activation="selu",
                      kernel_initializer=Orthogonal(np.sqrt(2)),
                      bias_initializer=Constant(0.0))(x)
        else:
            x = Conv2D(filters, kernel_size=8, strides=4, activation="selu")(net_inp)
            x = BatchNormalization()(x)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
            x = BatchNormalization()(x)
            x = Conv2D(filters*4, kernel_size=3, strides=1, activation="selu")(x)
            x = BatchNormalization()(x)
            x = Flatten()(x)
            x = Dense(512, activation="selu")(x)

    else:
        raise ValueError("Unknown Network Architecture \"{}\"".format(net_architecture))

    return x


class LogStdLayer(tf.keras.layers.Layer):
    def __init__(self, net_out_shape, **kwargs):
        super(LogStdLayer, self).__init__()
        self.net_out_shape = net_out_shape
        self.log_std = tf.Variable(tf.ones(self.net_out_shape) * -1.6, name='LOG_STD',
                                   trainable=True,
                                   constraint=lambda x: tf.clip_by_value(x, -20, 1))

    def call(self, inputs):
        return self.log_std

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'net_out_shape': self.net_out_shape,
        })
        return config
