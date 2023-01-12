import numpy as np
from ..misc.replay_buffer import FIFOBuffer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from ..misc.network_constructor import construct_network
import tensorflow as tf
from ..training_algorithms.agent_blueprint import Learner
import itertools
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
import time
from collections import deque
from tensorflow.keras.utils import plot_model

class MetaLearning:
    Name = "None"
    ParameterSpace = {}

    def __init__(self, action_shape, observation_shape, action_space, meta_learning_parameters, idx):
        self.action_space = action_space
        self.action_shape = action_shape

    def act(self):
        return

    def learning_step(self, _param1, _param2):
        return

    def reset(self):
        return

    def get_logs(self):
        return {}

    def prevent_checkpoint(self):
        return False

    @staticmethod
    def get_config(config_dict=None):
        if not config_dict:
            config_dict = MetaLearning.__dict__
        config_dict = {key: val for (key, val) in config_dict.items()
                       if not key.startswith('__')
                       and not callable(val)
                       and not type(val) is staticmethod
                       }
        return config_dict