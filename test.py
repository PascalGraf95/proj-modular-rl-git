import tensorflow as tf
import numpy as np
import time

np.random.seed(42)
tf.random.set_seed(42)


class Mooodel():
    def __init__(self):
        input_layer = tf.keras.Input(shape=(None, 2))
        lstm_out, hidden_state, cell_state = tf.keras.layers.LSTM(3, return_state=True)(input_layer)
        output = tf.keras.layers.Dense(2)(lstm_out)
        self.model = tf.keras.Model(inputs=input_layer, outputs=[output, hidden_state, cell_state])
        self.lstm = self.model.get_layer("lstm")
        print(self.lstm.units)

    def get_zero_initial_state(self, inputs):
        return [tf.zeros((2, 3)), tf.zeros((2, 3))]

    def get_initial_state(self, inputs):
        return self.initial_state

    def set_zero_initial_state(self):
        self.lstm.get_initial_state = lambda x: [tf.zeros((2, 3)), tf.zeros((2, 3))]

    def set_initial_state(self, states):
        self.initial_state = states
        self.lstm.get_initial_state = lambda x: states

    def __call__(self, inputs, states=None):
        """
        if states is None:
            self.lstm.get_initial_state = self.get_zero_initial_state

        else:
            self.initial_state = states
            self.lstm.get_initial_state = self.get_initial_state
        """

        return self.model(inputs)


if __name__ == '__main__':
    lstm_state = [np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)]
    #print(lstm_state)
    #print(lstm_state[0].shape)
    #print(tf.zeros(3).shape)
    #lstm_state[0][0] = np.zeros(3)
    #lstm_state[1][0] = np.zeros(3)
    print(lstm_state)

    mdl = Mooodel()
    time.sleep(3)
    x = np.random.rand(2, 1, 2).astype(np.float32)
    out, hidden_state, cell_state = mdl(x)
    print(np.mean(out))
    mdl.set_initial_state([tf.convert_to_tensor(lstm_state[0]), tf.convert_to_tensor(lstm_state[1])])
    out, hidden_state, cell_state = mdl(x)
    print(np.mean(out))
    mdl.set_zero_initial_state()
    out, hidden_state, cell_state = mdl(x)
    print(np.mean(out))
