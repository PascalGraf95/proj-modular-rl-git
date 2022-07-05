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
    '''state = [np.array([[0., 0., 0., 0., 1., 0., 0.5393357, 0., 0., 0., 0., 1., 0., 0.41818592, 0., 0., 0., 0.,
                        1., 0., 0.5442544, 0., 0., 0., 0., 1., 0., 0.32862315, 0., 0., 0., 0., 1., 0.,
                        0.8650164]], dtype=np.float32), np.array([[0.8650164]], dtype=np.float32)]
    state.append(np.array([[0.69]], dtype=np.float32))'''

    temp_buffer = deque(maxlen=3000)
    global_buffer = deque(maxlen=90)

    num_arms = 32

    arm_play_count = np.zeros(num_arms)
    empirical_mean = np.zeros(num_arms)

    for x in range(20):
        temp_buffer.append([np.random.randint(num_arms), np.array(np.random.rand())])
    for x in range(100):
        global_buffer.append(temp_buffer)

    for episode in global_buffer:
        for j, reward in episode:
            arm_play_count[j] += 1
            empirical_mean[j] += reward
    chosen_arm = np.argmax(empirical_mean + 1 * np.sqrt(1 / (arm_play_count + 1e-6)))
    print(chosen_arm)

    '''state = [tf.expand_dims(single_state, axis=1) for single_state in state]
    mdl = Mooodel()
    time.sleep(3)
    state_embedding = mdl.feature_extractor(state)[0]
    action_prediction = mdl.embedding_classifier([state_embedding, state_embedding])
    print(state_embedding)'''
