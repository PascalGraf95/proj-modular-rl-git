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
from tensorflow.keras.utils import plot_model


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

    def __init__(self, action_shape, observation_shapes,
                 action_space,
                 exploration_parameters,
                 training_parameters, idx):
        self.action_space = action_space
        self.action_shape = action_shape
        self.observation_shapes = observation_shapes
        self.index = idx
        self.device = '/cpu:0'

        #if self.index == 0:
        self.recurrent = training_parameters["Recurrent"]
        self.sequence_length = training_parameters["SequenceLength"]

        self.feature_space_size = exploration_parameters["FeatureSpaceSize"]
        #self.reward_scaling_factor = exploration_parameters["CuriosityScalingFactor"]
        self.reward_scaling_factor = exploration_parameters["ExplorationDegree"]
        self.normalize_observations = exploration_parameters["ObservationNormalization"]
        self.mse = MeanSquaredError()
        self.optimizer = Adam(exploration_parameters["LearningRate"])

        self.intrinsic_reward = 0
        self.loss = 0

        self.observation_deque = deque(maxlen=1000)
        self.observation_mean = 0
        self.observation_std = 1
        self.reward_deque = deque(maxlen=1000)
        self.reward_mean = 0
        self.reward_std = 1

        self.prediction_model, self.target_model = self.build_network()

    def build_network(self):
        with tf.device(self.device):
            # region Prediction Model
            if len(self.observation_shapes) == 1:
                if self.recurrent:
                    feature_input = Input((None, *self.observation_shapes[0]))
                else:
                    feature_input = Input(self.observation_shapes[0])
                x = feature_input
            else:
                feature_input = []
                for obs_shape in self.observation_shapes:
                    if self.recurrent:
                        # Add additional time dimensions if networks work with recurrent replay batches
                        feature_input.append(Input((None, *obs_shape)))
                    else:
                        feature_input.append(Input(obs_shape))
                x = Concatenate()(feature_input)
            x = Dense(32, activation="relu")(x)
            x = Dense(32, activation="relu")(x)
            x = Dense(self.feature_space_size, activation=None)(x)
            prediction_model = Model(feature_input, x, name="RND_PredictionModel")
            # endregion

            # region Target Model
            if len(self.observation_shapes) == 1:
                if self.recurrent:
                    feature_input = Input((None, *self.observation_shapes[0]))
                else:
                    feature_input = Input(self.observation_shapes[0])
                x = feature_input
            else:
                feature_input = []
                for obs_shape in self.observation_shapes:
                    if self.recurrent:
                        # Add additional time dimensions if networks work with recurrent replay batches
                        feature_input.append(Input((None, *obs_shape)))
                    else:
                        feature_input.append(Input(obs_shape))
                x = Concatenate()(feature_input)
            x = Dense(32, activation="relu")(x)
            x = Dense(32, activation="relu")(x)
            x = Dense(self.feature_space_size, activation=None)(x)
            target_model = Model(feature_input, x, name="RND_TargetModel")
            # endregion

            # Model plots
            try:
                plot_model(prediction_model, "plots/RND_PredictionModel.png", show_shapes=True)
                plot_model(target_model, "plots/RND_TargetModel.png", show_shapes=True)
            except ImportError:
                print("Could not create model plots for RND.")

            # Summaries
            prediction_model.summary()
            target_model.summary()
            # endregion

            '''# List with two dictionaries in it, one for each network.
            network_parameters = [{}, {}]
            # region --- Prediction Model ---
            # This model tries to mimic the target model for every state in the environment
            # - Network Name -
            network_parameters[0]['NetworkName'] = "RND_PredictionModel"
            # - Network Architecture-
            network_parameters[0]['VectorNetworkArchitecture'] = "SmallDense"
            network_parameters[0]['VisualNetworkArchitecture'] = "CNN"
            network_parameters[0]['Filters'] = 32
            network_parameters[0]['Units'] = 32
            network_parameters[0]['TargetNetwork'] = False
            # - Input / Output / Initialization -
            network_parameters[0]['Input'] = self.observation_shapes
            network_parameters[0]['Output'] = [self.feature_space_size]
            network_parameters[0]['OutputActivation'] = [None]
            # - Recurrent Parameters -
            network_parameters[0]['Recurrent'] = False
            # endregion

            # region --- TargetModel ---
            network_parameters[1] = network_parameters[0].copy()
            network_parameters[1]['NetworkName'] = "RND_TargetModel"

            prediction_model = construct_network(network_parameters[0], plot_network_model=True)
            target_model = construct_network(network_parameters[1], plot_network_model=True)'''

            return prediction_model, target_model

    def learning_step(self, replay_batch):
        # region --- Batch Reshaping ---
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
        # endregion

        with tf.device(self.device):
            with tf.GradientTape() as tape:
                target_features = self.target_model(next_state_batch)
                prediction_features = self.prediction_model(next_state_batch)
                self.loss = self.mse(target_features, prediction_features)
            # Calculate Gradients and apply the weight updates to the prediction model.
            grad = tape.gradient(self.loss, self.prediction_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grad, self.prediction_model.trainable_weights))
        return

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = RandomNetworkDistillation.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        if not len(decision_steps.obs[0]):
            self.reward_deque.append(0)
            return 0
        current_state = decision_steps.obs

        # region Lifelong Novelty Module
        if self.normalize_observations:
            # Track observation metrics
            for idx, state in enumerate(current_state):
                state = np.mean(state)
                self.observation_deque.append(state)
            self.observation_mean = np.mean(self.observation_deque)
            self.observation_std = np.std(self.observation_deque)
            # Normalize observation values
            current_state -= self.observation_mean
            current_state /= self.observation_std
            current_state = np.clip(current_state, -5, 5)

        # Calculate the features for the current state with the target and the prediction model.
        if self.recurrent:
            # Add additional time dimension if recurrent networks are used
            current_state = [tf.expand_dims(state, axis=1) for state in current_state]
            target_features = self.target_model(current_state)[0]
            prediction_features = self.prediction_model(current_state)[0]
        else:
            # Calculate the features for the current state with the target and the prediction model.
            target_features = self.target_model(current_state)
            prediction_features = self.prediction_model(current_state)

        # The rnd reward is the L2 error between target and prediction features summed over all features.
        self.intrinsic_reward = tf.math.sqrt(tf.math.reduce_sum(
            tf.math.square(target_features - prediction_features), axis=-1)).numpy()
        self.intrinsic_reward = np.clip(self.intrinsic_reward[0], -1, 100)

        # Calculate the running standard deviation and mean of the rnd rewards to normalize it.
        self.reward_deque.append(self.intrinsic_reward)
        self.reward_std = np.std(self.reward_deque)
        self.reward_mean = np.mean(self.reward_deque)

        # Normalize reward
        if self.reward_std:
            self.intrinsic_reward /= self.reward_std

        # Scale intrinsic reward by exploration degree
        self.intrinsic_reward *= self.reward_scaling_factor

        return self.intrinsic_reward

    def get_logs(self):
        return {"Exploration/RNDLoss": self.loss,
                "Exploration/Reward_Act{:02d}_{:.4f}".format(self.index, self.reward_scaling_factor): self.intrinsic_reward}

    def prevent_checkpoint(self):
        return False
