import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import itertools
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
import time
from collections import deque
from tensorflow.keras.utils import plot_model

np.random.seed(42)
tf.random.set_seed(42)


class Mooodel():
    def __init__(self):
        self.observation_shapes = [(35,)]
        self.feature_space_size = 16

        # Override observation shapes
        modified_observation_shapes = []
        for obs_shape in self.observation_shapes:
            modified_observation_shapes.append(obs_shape)
        modified_observation_shapes.append((1,))
        modified_observation_shapes.append((1,))
        self.observation_shapes = modified_observation_shapes

        # region Feature Extractor
        if len(self.observation_shapes) == 1:
            feature_input = Input((None, *self.observation_shapes[0]))
            x = feature_input
        else:
            feature_input = []
            for obs_shape in self.observation_shapes:
                # obs_shape = (obs_shape,)
                feature_input.append(Input((None, *obs_shape)))
            x = Concatenate()(feature_input)
        x = Dense(32, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        x = Dense(16, activation="relu")(x)

        self.feature_extractor = Model(feature_input, x, name="ENM Feature Extractor")
        # endregion

        # region Classifier
        #current_state_features = Input(self.feature_space_size)
        #next_state_features = Input(self.feature_space_size)

        current_state_features = Input((None, *(self.feature_space_size,)))
        next_state_features = Input((None, *(self.feature_space_size,)))

        x = Concatenate(axis=-1)([current_state_features, next_state_features])
        x = Dense(128, 'relu')(x)
        x = Dense(2, 'tanh')(x)  # CONT
        self.embedding_classifier = Model([current_state_features, next_state_features], x, name="ENM Classifier")

        # Summaries
        self.feature_extractor.summary()
        self.embedding_classifier.summary()
        # endregion

if __name__ == '__main__':
    temp_a = np.array([[1,1],[1,2],[1,5],[3,4],[0,2]])
    temp_a = np.expand_dims(temp_a, axis=2)

    gamma = np.array([10,10,10,10,10])
    a = 1
    print('debug')
