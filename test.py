import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LSTM
import numpy as np
import time
import ray
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)
np.random.seed(42)
tf.random.set_seed(42)


@ray.remote
class Mooodel():
    def __init__(self):
        input_layer = tf.keras.Input(shape=2)
        x = Dense(1)(input_layer)
        # x = tf.keras.layers.LSTM(4, return_state=False)(x)
        # x = tf.keras.layers.Dense(2)(x)
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)

    def get_model(self):
        return self.model

    def predict_with_model(self, x):
        return self.model(x)

    """
    def get_zero_initial_state(self, inputs):
        return [tf.zeros((2, 3)), tf.zeros((2, 3))]

    def get_initial_state(self, inputs):
        return self.initial_state

    def set_zero_initial_state(self):
        self.lstm.get_initial_state = lambda x: [tf.zeros((2, 3)), tf.zeros((2, 3))]

    def set_initial_state(self, states):
        self.initial_state = states
        self.lstm.get_initial_state = lambda x: states
    """


if __name__ == '__main__':
    ray.init()

    model = Mooodel.remote()
    time.sleep(3)

    for i in range(100):
        x = np.random.rand(2, 2).astype(np.float32)
        out = model.predict_with_model.remote(x)
        print(str(i) + ": " + str(ray.get(out)))
        time.sleep(0.5)
