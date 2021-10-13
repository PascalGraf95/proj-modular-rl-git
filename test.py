import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import ray
import time
import numpy as np
#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
#    try:
#        tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
#    except RuntimeError as e:
#        print(e)

@ray.remote
class Actor:
    def __init__(self):
        with tf.device('/cpu:0'):
            self.construct_model()

    def construct_model(self):
        input_layer = tf.keras.layers.Input((None, 2))
        x = tf.keras.layers.Dense(16)(input_layer)
        x = tf.keras.layers.Dense(16)(x)
        x = tf.keras.layers.LSTM(5, return_sequences=True)(x)
        x = tf.keras.layers.Dense(2)(x)
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

    def print_layers(self):
        for layer in self.model.layers:
            if "lstm" in layer.name:
                print(layer.reset_states())

    def predict(self, state):
        with tf.device('/cpu:0'):
            return self.model(state)


@ray.remote(num_gpus=1)
class Learner:
    def __init__(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        self.construct_model()
        self.optimizer = tf.keras.optimizers.Adam()

    def construct_model(self):
        input_layer = tf.keras.layers.Input(1)
        x = tf.keras.layers.Dense(16)(input_layer)
        x = tf.keras.layers.Dense(3)(x)
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)

    def predict(self, state):
        return self.model(state)

    def train(self):
        s = np.random.random((2, 1))
        with tf.GradientTape() as tape:
            x = self.predict(s)
            loss = tf.losses.mse(np.random.random((2, 3)), x)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


if __name__ == '__main__':
    ray.init()
    actors = [Actor.remote() for i in range(1)]
    learner = Learner.remote()

    for actor in actors:
        actor.print_layers.remote()

    for i in range(1):
        for i in range(2):
            # batch_size, time_steps, obs
            s = np.random.random((1, 3, 2))
            res = [actor.predict.remote(s) for actor in actors]
            print("ACTOR", ray.get(res))
            #s = np.random.random((1, 1))
            #res = learner.predict.remote(s)
            #print("LEARNER:", ray.get(res))
            time.sleep(1)
        #loss = learner.train.remote()
        #print("LOSS:", ray.get(loss))
        # time.sleep(5)