import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import ray
import time
import numpy as np
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
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
        x, self.state_h, self.state_c = tf.keras.layers.LSTM(5, return_sequences=True, return_state=True)(x)
        x = tf.keras.layers.Dense(2)(x)
        self.model = tf.keras.Model(inputs=input_layer, outputs=x)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.summary()

    def print_layers(self):
        for layer in self.model.layers:
            if "lstm" in layer.name:
                print(layer.reset_states())

    def print_states(self):
        print(self.state_h, self.state_c)

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
    normal = tfp.distributions.Normal(-1.6182493, 0.002102602)
    print(normal.parameters)
    data = normal.sample(1, 10000)
    print(data.shape)
    hx, hy, _ = plt.hist(data, bins=50, color="lightblue")
    plt.xlim(-1.6182493-0.02, -1.6182493+0.02)

    plt.title(r'Normal distribution')
    plt.grid()
    # plt.show()
    # x = -1.617729
    x = -1.4182493
    d = normal.log_prob(x) / normal.log_prob(-1.6182493) # 2.9427497
    print(d)
    """
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
            [actor.print_states.remote() for actor in actors]
            #s = np.random.random((1, 1))
            #res = learner.predict.remote(s)
            #print("LEARNER:", ray.get(res))
            time.sleep(1)
        #loss = learner.train.remote()
        #print("LOSS:", ray.get(loss))
        # time.sleep(5)
    """
