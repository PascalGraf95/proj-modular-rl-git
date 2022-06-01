#!/usr/bin/env python

from enum import Enum
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input, BatchNormalization, Dropout, Add, Subtract, \
    Lambda, UpSampling2D, Reshape, LSTM, GlobalMaxPool2D, TimeDistributed
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
ImageShapeLength = 4
VectorShapeLength = 2


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


"""
This file contains methods to construct neural networks in Tensorflow given the type of network, the number of units
or filters as well as the desired input and output shapes.
"""


def construct_network(network_parameters, plot_network_model=False):
    """Constructs a neural network for any Reinforcement Learning algorithm with arbitrary input and output shapes
    and a defined type of network body (see NetworkArchitecture enum above)."""

    # Distinguish Dense from NoisyDense Layer (Only for DQN)
    global Dense
    if network_parameters.get("NoisyNetworks"):
        Dense = NoisyDense
    else:
        Dense = NormalDense

    # region --- Input ---
    # Construct one input layer per input component.
    network_input = build_network_input(network_parameters.get('Input'), network_parameters)
    network_branches = network_input

    # Depending on the number and types of the inputs either keep them separate or connect them here.
    # Multiple image inputs will be stacked, multiple vector inputs will be concatenated, a mixture of vector
    # and image inputs will be processed with separate network bodies and connected afterwards.
    x = connect_branches(network_branches, network_parameters)
    # endregion

    # region --- Body ---
    # Construct the network body/bodies
    network_body, hidden_state, cell_state = build_network_body(x, network_parameters)
    # endregion

    # region --- Output ---
    # Construct the network output
    network_output = build_network_output(network_body, network_parameters)
    # endregion

    # region --- Model Construction ---
    if network_parameters.get("Recurrent") and network_parameters.get("ReturnStates"):
        model = Model(inputs=network_input, outputs=[network_output, hidden_state, cell_state],
                      name=network_parameters.get("NetworkName"))
    else:
        model = Model(inputs=network_input, outputs=network_output, name=network_parameters.get("NetworkName"))
    # endregion

    # region --- Model Plot ---
    if plot_network_model:
        model.summary()
        try:
            plot_model(model, "plots/"+network_parameters.get("NetworkName") + ".png", show_shapes=True)
        except ImportError:
            print("Could not create a model plot for {}.".format(network_parameters.get("NetworkType")))
    # endregion

    if network_parameters.get('TargetNetwork'):
        target_model = clone_model(model)
        return model, target_model
    return model


def build_network_input(network_input_shapes, network_parameters):
    """ Builds the input layers for an arbitrary number of inputs with arbitrary shape."""
    network_input = []
    # TODO: Add intrinsic reward and exploration policy index j as network inputs
    # Iterate through all input components
    for input_shapes in network_input_shapes:
        # In case of a recurrent neural network, the input shape of vector observation changes from 2 to 3,
        # for an image component from 4 to 5.
        if network_parameters["Recurrent"]:
            if type(input_shapes) == int:
                input_shapes = (input_shapes,)
            # For stateful recurrent neural networks the batch size needs to be predefined and constant.
            x = Input((None, *input_shapes), batch_size=network_parameters["BatchSize"])
            global ImageShapeLength
            global VectorShapeLength
            ImageShapeLength = 5
            VectorShapeLength = 3
        else:
            x = Input(input_shapes)
        network_input.append(x)
    return network_input


def connect_branches(network_input, network_parameters):
    """Connects multiple vector XOR image branches via Concatenate layers"""
    branch_shape_lengths = [len(x.shape) for x in network_input]
    input_types = None
    # Check if there is more than one input branch
    if len(branch_shape_lengths) > 1:
        # Do all input branches have the same shape, i.e. are they all images or all vectors.
        if len(set(branch_shape_lengths)) == 1:
            # If all inputs are images or vectors, just Concatenate them
            if branch_shape_lengths[0] == ImageShapeLength or branch_shape_lengths[0] == VectorShapeLength:
                return [Concatenate(axis=-1)(network_input)]
        # Otherwise, the inputs are mixed between images and vectors
        else:
            # If the Vec2Img-option is enabled, Vectors will be reshaped into images and resized to enable
            # concatenation
            if network_parameters.get("Vec2Img"):
                # Vector Branches will be reshaped and resized
                image_branches = []
                for net_input in [x for x in network_input if len(x.shape) == VectorShapeLength]:
                    x = Reshape((1, 1, net_input.shape[-1]))(net_input)
                    x = UpSampling2D(size=(84, 84))(x)
                    image_branches.append(x)
                # Image Branches
                for net_input in [x for x in network_input if len(x.shape) == ImageShapeLength]:
                    image_branches.append(net_input)
                if len(image_branches) > 1:
                    return [Concatenate(axis=-1)(image_branches)]
            else:
                # Vector Branches will be concatenated
                vector_branches = []
                for net_input in [x for x in network_input if len(x.shape) == VectorShapeLength]:
                    vector_branches.append(net_input)
                if len(vector_branches) > 1:
                    vector_branches = Concatenate(axis=-1)(vector_branches)
                else:
                    vector_branches = vector_branches[0]
                # Image Branches will be concatenated
                image_branches = []
                for net_input in [x for x in network_input if len(x.shape) == ImageShapeLength]:
                    image_branches.append(net_input)
                if len(image_branches) > 1:
                    image_branches = Concatenate(axis=-1)(image_branches)
                else:
                    image_branches = image_branches[0]
                # Image and Vector branches returned separately
                return [image_branches, vector_branches]
    else:
        # If there is only one branch, return
        return network_input


def build_network_body(network_input, network_parameters):
    """Builds the actual network body based on the network type keyword and the number of units/filters.
    If the network branches have not been connected yet, each branch receives its own network body. Their outputs
    are connected afterwards."""
    network_branches = []
    units, filters = 0, 0
    hidden_state, cell_state = None, None
    # Construct one network body for each network branch
    for net_input in network_input:
        # In case of Vector input utilize the vector architecture and units
        if len(net_input.shape) == VectorShapeLength:
            net_architecture = network_parameters.get("VectorNetworkArchitecture")
            units = network_parameters.get("Units")
        else:
            # In case of Image input utilize the visual architecture and filters
            net_architecture = network_parameters.get("VisualNetworkArchitecture")
            filters = network_parameters.get("Filters")

        # Distinguish between Recurrent and Feed Forward neural networks.
        # Recurrent network bodies return the hidden and cell state additionally to the last layer output x
        if network_parameters.get("Recurrent") and network_parameters.get("ReturnStates"):
            network_branch, hidden_state, cell_state = get_network_component(net_input, net_architecture,
                                                                             network_parameters,
                                                                             units=units, filters=filters,
                                                                             multi_branch=len(network_input) > 1)
        else:
            network_branch = get_network_component(net_input, net_architecture,
                                                   network_parameters, units=units, filters=filters,
                                                   multi_branch=len(network_input) > 1)
        network_branches.append(network_branch)
    # If there are multiple branches, connect them to one.
    network_body = connect_branches(network_branches, network_parameters)
    # If there have been multiple branches before, add one Dense layer to the end where all information are processed
    # together.
    if len(network_branches) > 1:
        if network_parameters.get("Recurrent") and network_parameters.get("ReturnStates"):
            network_body, hidden_state, cell_state = get_network_component(network_body[0], "SingleDense",
                                                                           network_parameters,
                                                                           units=network_parameters.get("Units")*5)
            network_body = [network_body]
        else:
            network_body = [get_network_component(network_body[0], "SingleDense",
                                                  network_parameters, units=network_parameters.get("Units")*5)]
    return network_body[0], hidden_state, cell_state


def build_network_output(net_in, network_parameters):
    """Builds one or multiple network outputs. Supports dueling networks and special kernel initialization."""
    network_output = []
    # Iterate through all outputs in the parameters dictionary
    for net_out_shape, net_out_act in zip(network_parameters.get('Output'),
                                          network_parameters.get('OutputActivation')):
        if net_out_shape:
            if type(net_out_shape) == tuple:
                net_out_shape = net_out_shape[0]
            # Dueling Networks (only for DQN) have a special kind of output configuration.
            if network_parameters.get('DuelingNetworks'):
                y = Dense(network_parameters.get('Units'), activation="selu")(net_in)
                z = Dense(network_parameters.get('Units'), activation="selu")(net_in)

                y = Dense(1, name="value")(y)
                z = Dense(net_out_shape, activation=net_out_act, name="advantage")(z)
                z_mean = Lambda(lambda a: K.mean(a[:, 1:], axis=1, keepdims=True), name="mean")(z)
                net_out = Add()([y, z])
                net_out = Subtract(name="action")([net_out, z_mean])
            else:
                # Otherwise, just construct one Dense layer with the specified number of units and activation.
                # Some algorithms perform better with a special kind of kernel initializer.
                if network_parameters.get('KernelInitializer') == "RandomUniform":
                    net_out = Dense(net_out_shape, activation=net_out_act,
                                    kernel_initializer=RandomUniform(minval=-3e-3, maxval=3e-3),
                                    bias_initializer=RandomUniform(minval=-3e-3, maxval=3e-3))(net_in)
                else:
                    net_out = Dense(net_out_shape, activation=net_out_act)(net_in)
            network_output.append(net_out)
        else:
            network_output.append(net_in)
    return network_output


def connect_network_branches(network_branches, network_parameters):
    if len(network_branches) == 1:
        return network_branches[0]

    x = Concatenate(axis=-1)(network_branches)
    x = Dense(network_parameters.get('Filters')*5)(x)
    return x


def get_network_component(net_inp, net_architecture, network_parameters, units=32, filters=32, multi_branch=False):
    """Returns a special type of network architecture based on a keyword and the given input."""
    hidden_state, cell_state = None, None
    if net_architecture == NetworkArchitecture.NONE.value:
        x = net_inp

    # region --- Dense Architectures ---
    elif net_architecture == NetworkArchitecture.SINGLE_DENSE.value:
        if network_parameters["Recurrent"]:
            if network_parameters.get("ReturnStates"):
                x, hidden_state, cell_state = LSTM(units, return_sequences=network_parameters["ReturnSequences"],
                                                   stateful=network_parameters["Stateful"], return_state=True)(net_inp)
            else:
                x = LSTM(units, return_sequences=network_parameters["ReturnSequences"],
                         stateful=network_parameters["Stateful"])(net_inp)
        else:
            x = Dense(units, activation='selu')(net_inp)

    elif net_architecture == NetworkArchitecture.SMALL_DENSE.value:
        x = Dense(units, activation='selu')(net_inp)
        if network_parameters["Recurrent"]:
            if not multi_branch:
                if network_parameters.get("ReturnStates"):
                    x, hidden_state, cell_state = LSTM(units, return_sequences=network_parameters["ReturnSequences"],
                                                       stateful=network_parameters["Stateful"], return_state=True)(x)
                else:
                    x = LSTM(units, return_sequences=network_parameters["ReturnSequences"],
                             stateful=network_parameters["Stateful"])(x)
        else:
            x = Dense(units, activation='selu')(x)

    elif net_architecture == NetworkArchitecture.DENSE.value:
        x = Dense(units, activation='selu')(net_inp)
        x = Dense(units, activation='selu')(x)
        if network_parameters["Recurrent"]:
            if not multi_branch:
                if network_parameters.get("ReturnStates"):
                    x, hidden_state, cell_state = LSTM(2*units, return_sequences=network_parameters["ReturnSequences"],
                                                       stateful=network_parameters["Stateful"], return_state=True)(x)
                else:
                    x = LSTM(2*units, return_sequences=network_parameters["ReturnSequences"],
                             stateful=network_parameters["Stateful"])(x)
        else:
            x = Dense(2*units, activation='selu')(x)

    elif net_architecture == NetworkArchitecture.LARGE_DENSE.value:
        x = Dense(units, activation='selu')(net_inp)
        x = Dense(units, activation='selu')(x)
        x = Dense(2*units, activation='selu')(x)
        if network_parameters["Recurrent"]:
            if not multi_branch:
                if network_parameters.get("ReturnStates"):
                    x, hidden_state, cell_state = LSTM(2*units, return_sequences=network_parameters["ReturnSequences"],
                                                       stateful=network_parameters["Stateful"], return_state=True)(x)
                else:
                    x = LSTM(2*units, return_sequences=network_parameters["ReturnSequences"],
                             stateful=network_parameters["Stateful"])(x)
        else:
            x = Dense(2*units, activation='selu')(x)
    # endregion

    # region --- CNN Architectures ---
    elif net_architecture == NetworkArchitecture.CNN.value:
        x = Conv2D(filters, kernel_size=8, strides=4, activation="selu")(net_inp)
        x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
        x = Conv2D(filters*2, kernel_size=3, strides=1, activation="selu")(x)
        if network_parameters["Recurrent"]:
            x = Reshape((-1, x.shape[2]*x.shape[3]*x.shape[4]))(x)
            if not multi_branch:
                if network_parameters.get("ReturnStates"):
                    x, hidden_state, cell_state = LSTM(filters*10, return_sequences=network_parameters["ReturnSequences"],
                                                       stateful=network_parameters["Stateful"], return_state=True)(x)
                else:
                    x = LSTM(filters*10, return_sequences=network_parameters["ReturnSequences"],
                             stateful=network_parameters["Stateful"])(x)
        else:
            x = Flatten()(x)
            x = Dense(filters*10, activation="selu")(x)
            print(x.shape)

    elif net_architecture == NetworkArchitecture.CNN_ICM.value:
        x = Conv2D(filters, kernel_size=3, strides=2, padding="same", activation="selu")(net_inp)
        x = Conv2D(filters, kernel_size=3, strides=2, padding="same", activation="selu")(x)
        x = Conv2D(filters, kernel_size=3, strides=2, padding="same", activation="selu")(x)
        x = Conv2D(filters, kernel_size=3, strides=2, padding="same", activation="selu")(x)
        x = Flatten()(x)

    elif net_architecture == NetworkArchitecture.CNN_BATCHNORM.value:
        x = Conv2D(filters, kernel_size=8, strides=4, activation="selu")(net_inp)
        x = BatchNormalization()(x)
        x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters*2, kernel_size=3, strides=1, activation="selu")(x)
        x = BatchNormalization()(x)

        if network_parameters["Recurrent"]:
            x = Reshape((-1, x.shape[2]*x.shape[3]*x.shape[4]))(x)
            if not multi_branch:
                if network_parameters.get("ReturnStates"):
                    x, hidden_state, cell_state = LSTM(512, return_sequences=network_parameters["ReturnSequences"],
                                                       stateful=network_parameters["Stateful"], return_state=True)(x)
                else:
                    x = LSTM(512, return_sequences=network_parameters["ReturnSequences"],
                             stateful=network_parameters["Stateful"])(x)
        else:
            x = Flatten()(x)
            x = Dense(512, activation="selu")(x)

    elif net_architecture == NetworkArchitecture.SHALLOW_CNN.value:
        pass

    elif net_architecture == NetworkArchitecture.DEEP_CNN.value:
        x = Conv2D(filters, kernel_size=8, strides=4, activation="selu")(net_inp)
        x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
        x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
        x = Conv2D(filters*4, kernel_size=3, strides=1, activation="selu")(x)

        if network_parameters["Recurrent"]:
            x = Reshape((-1, x.shape[2]*x.shape[3]*x.shape[4]))(x)
            if not multi_branch:
                if network_parameters.get("ReturnStates"):
                    x, hidden_state, cell_state = LSTM(512, return_sequences=network_parameters["ReturnSequences"],
                                                       stateful=network_parameters["Stateful"], return_state=True)(x)
                else:
                    x = LSTM(512, return_sequences=network_parameters["ReturnSequences"],
                             stateful=network_parameters["Stateful"])(x)
        else:
            x = Flatten()(x)
            x = Dense(512, activation="selu")(x)

    elif net_architecture == NetworkArchitecture.DEEP_CNN_BATCHNORM.value:
        x = Conv2D(filters, kernel_size=8, strides=4, activation="selu")(net_inp)
        x = BatchNormalization()(x)
        x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters*2, kernel_size=4, strides=2, activation="selu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters*4, kernel_size=3, strides=1, activation="selu")(x)
        x = BatchNormalization()(x)

        if network_parameters["Recurrent"]:
            x = Reshape((-1, x.shape[2]*x.shape[3]*x.shape[4]))(x)
            if not multi_branch:
                if network_parameters.get("ReturnStates"):
                    x, hidden_state, cell_state = LSTM(512, return_sequences=network_parameters["ReturnSequences"],
                                                       stateful=network_parameters["Stateful"], return_state=True)(x)
                else:
                    x = LSTM(512, return_sequences=network_parameters["ReturnSequences"],
                             stateful=network_parameters["Stateful"])(x)
        else:
            x = Flatten()(x)
            x = Dense(512, activation="selu")(x)
    # endregion

    else:
        raise ValueError("Unknown Network Architecture \"{}\"".format(net_architecture))
    if network_parameters.get("ReturnStates") and network_parameters.get("Recurrent"):
        return x, hidden_state, cell_state
    return x

