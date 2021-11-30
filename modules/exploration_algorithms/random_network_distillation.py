import numpy as np
from ..misc.replay_buffer import FIFOBuffer
from .exploration_algorithm_blueprint import ExplorationAlgorithm
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


class RandomNetworkDistillation(ExplorationAlgorithm):
    """
    Basic implementation of Random Network Distillation (RND)
    """
    Name = "RandomNetworkDistillation"
    ActionAltering = False
    IntrinsicReward = True

    ParameterSpace = {
        "FeatureSpaceSize": int,
        "CuriosityScalingFactor": float,
        "LearningRate": float,
    }

    def __init__(self, action_shape, observation_shapes, action_space, exploration_parameters, training_parameters):
        self.action_space = action_space
        self.action_shape = action_shape
        self.observation_shapes = observation_shapes
        self.recurrent = training_parameters["Recurrent"]
        self.sequence_length = training_parameters["SequenceLength"]

        self.feature_space_size = exploration_parameters["FeatureSpaceSize"]
        self.reward_scaling_factor = exploration_parameters["CuriosityScalingFactor"]
        self.mse = MeanSquaredError()
        self.optimizer = Adam(exploration_parameters["LearningRate"])

        self.max_intrinsic_reward = 0
        self.mean_intrinsic_reward = 0
        self.loss = 0
        self.calculated_reward = False

        self.observation_deque = deque(maxlen=1000)
        self.observation_mean = 0
        self.observation_std = 1
        self.reward_deque = deque(maxlen=1000)
        self.reward_mean = 0
        self.reward_std = 1

        self.prediction_model, self.target_model = self.build_network()

    def build_network(self):
        with tf.device('/cpu:0'):
            # Prediction Model: tries to mimic the target model
            input_layers = []
            for obs_shape in self.observation_shapes:
                input_layers.append(Input(obs_shape))
            if len(input_layers) > 1:
                x = Concatenate()(input_layers)
            else:
                x = input_layers[0]
            x = Dense(64, activation="elu")(x)
            x = Dense(64, activation="elu")(x)
            x = Dense(self.feature_space_size, activation="linear")(x)
            prediction_model = Model(input_layers, x, name="RND Prediction Model")

            # Target Model: Randomly Initialized
            input_layers = []
            for obs_shape in self.observation_shapes:
                input_layers.append(Input(obs_shape))
            if len(input_layers) > 1:
                x = Concatenate()(input_layers)
            else:
                x = input_layers[0]
            x = Dense(64, activation="elu")(x)
            x = Dense(64, activation="elu")(x)
            x = Dense(self.feature_space_size, activation="linear")(x)
            target_model = Model(input_layers, x, name="RND Target Model")

            return prediction_model, target_model

    def learning_step(self, replay_batch):
        return

    def get_intrinsic_reward(self, replay_batch):
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)

        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes,
                                                               self.action_shape)
        if np.any(np.isnan(action_batch)):
            return replay_batch

        # Normalize the observations
        for state in state_batch:
            state -= self.observation_mean
            state /= self.observation_std
            state = np.clip(state, -5, 5)

        self.calculated_reward = True

        # Predict features with the target and prediction model
        with tf.device('/cpu:0'):
            with tf.GradientTape() as tape:
                target_features = self.target_model(state_batch)
                prediction_features = self.prediction_model(state_batch)
                self.loss = self.mse(target_features, prediction_features)
            # Calculate Gradients and apply the weight updates
            grad = tape.gradient(self.loss, self.prediction_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grad, self.prediction_model.trainable_weights))

        # Calculate the intrinsic reward by MSE between prediction and actual next state feature
        intrinsic_reward = tf.math.reduce_sum(
            tf.math.abs(target_features - prediction_features), axis=-1).numpy()

        # Update the running average and standard deviation of the past rewards
        if self.recurrent:
            for reward_sequence in intrinsic_reward:
                for reward in reward_sequence:
                    self.reward_deque.append(reward)
            self.reward_std = np.std(self.reward_deque)
        else:
            for reward in intrinsic_reward:
                self.reward_deque.append(reward)

        # Normalize the intrinsic reward
        intrinsic_reward /= self.reward_std
        # Scale the reward
        intrinsic_reward *= self.reward_scaling_factor
        self.max_intrinsic_reward = np.max(intrinsic_reward)
        self.mean_intrinsic_reward = np.mean(intrinsic_reward)

        # Replace the original reward in the replay batch
        if self.recurrent:
            for seq_idx, reward_sequence in enumerate(intrinsic_reward):
                for idx, rew in enumerate(reward_sequence):
                    replay_batch[seq_idx][idx]["reward"] = \
                        replay_batch[seq_idx][idx]["reward"]*(1-self.reward_scaling_factor)+rew
        else:
            for idx, rew in enumerate(intrinsic_reward):
                replay_batch[idx]["reward"] = \
                    replay_batch[idx]["reward"]*(1-self.reward_scaling_factor)+rew
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = RandomNetworkDistillation.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, index=None):
        if index or not len(decision_steps):
            return None
        for idx, state in enumerate(decision_steps.obs):
            state = np.mean(state)
            self.observation_deque.append(state)
        self.observation_mean = np.mean(self.observation_deque)
        self.observation_std = np.std(self.observation_deque)

    def get_logs(self, idx):
        if self.calculated_reward:
            self.calculated_reward = False
            return {"Exploration/RNDLoss": self.loss,
                    "Exploration/MaxIntrinsicReward": self.max_intrinsic_reward,
                    "Exploration/MeanIntrinsicReward": self.mean_intrinsic_reward}
        else:
            return None


    def prevent_checkpoint(self):
        return False
