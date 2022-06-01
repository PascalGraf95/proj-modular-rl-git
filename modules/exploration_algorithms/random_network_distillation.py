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

    def __init__(self, action_shape, observation_shapes,
                 action_space,
                 exploration_parameters,
                 training_parameters, idx):
        self.action_space = action_space
        self.action_shape = action_shape
        self.observation_shapes = observation_shapes
        self.index = idx
        self.device = '/cpu:0'

        if self.index == 0:
            self.recurrent = training_parameters["Recurrent"]
            self.sequence_length = training_parameters["SequenceLength"]

            self.feature_space_size = exploration_parameters["FeatureSpaceSize"]
            self.reward_scaling_factor = exploration_parameters["CuriosityScalingFactor"]
            self.normalize_observations = exploration_parameters["ObservationNormalization"]
            self.mse = MeanSquaredError()
            self.optimizer = Adam(exploration_parameters["LearningRate"])

            self.max_intrinsic_reward = 0
            self.mean_intrinsic_reward = 0
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
            # List with two dictionaries in it, one for each network.
            network_parameters = [{}, {}]
            # region --- Prediction Model ---
            # This model tries to mimic the target model for every state in the environment
            # - Network Name -
            network_parameters[0]['NetworkName'] = "RND_PredictionModel"
            # - Network Architecture-
            network_parameters[0]['VectorNetworkArchitecture'] = "SmallDense"
            network_parameters[0]['VisualNetworkArchitecture'] = "CNN"
            network_parameters[0]['Filters'] = 32
            network_parameters[0]['Units'] = 64
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
            target_model = construct_network(network_parameters[1], plot_network_model=True)

            return prediction_model, target_model

    def learning_step(self, replay_batch):
        """This learning step is directly integrated into the intrinsic reward calculation due to efficiency."""
        return

    def get_intrinsic_reward(self, replay_batch):
        # region --- RECURRENT ---
        if self.recurrent:
            # region --- Batch Reshaping ---
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)

            if np.any(np.isnan(action_batch)):
                return replay_batch
            # endregion

            intrinsic_reward = []
            # Loop through state sequences
            for idx, next_state_sequence in enumerate(next_state_batch[0]):
                # TODO: Check if next state observations are normalized and afterwards used
                # Normalize the observations
                if self.normalize_observations:
                    for state in next_state_batch:
                        state -= self.observation_mean
                        state /= self.observation_std
                        state = np.clip(state, -5, 5)

                # Casting current sequence as list (MUST be done)
                next_state_sequence = [next_state_sequence]

                # region --- Feature Calculation and Learning Step ---
                with tf.device(self.device):
                    # Calculate the features for the current next state batch with the target and the prediction model.
                    # Then calculate the Mean Squared Error between them.
                    with tf.GradientTape() as tape:
                        target_features = self.target_model(next_state_sequence)
                        prediction_features = self.prediction_model(next_state_sequence)
                        self.loss = self.mse(target_features, prediction_features)

                    # Calculate Gradients and apply the weight updates to the prediction model.
                    grad = tape.gradient(self.loss, self.prediction_model.trainable_weights)
                    self.optimizer.apply_gradients(zip(grad, self.prediction_model.trainable_weights))
                # endregion

                # The intrinsic reward is the L2 error between target and prediction features summed over all features.
                intrinsic_reward.append(tf.math.sqrt(
                    tf.math.reduce_sum(tf.math.square(target_features - prediction_features), axis=-1)).numpy())

            # Cast as array
            intrinsic_reward = np.array(intrinsic_reward)

            # Calculate the standard deviation of the intrinsic rewards to normalize them.
            for reward_sequence in intrinsic_reward:
                for reward in reward_sequence:
                    self.reward_deque.append(reward)
            self.reward_std = np.std(self.reward_deque)

            # Normalize the intrinsic reward by the standard deviation
            intrinsic_reward /= self.reward_std
            # Scale the reward
            intrinsic_reward *= self.reward_scaling_factor

            # region --- Store Max and Mean intrinsic rewards
            self.max_intrinsic_reward = np.max(intrinsic_reward)
            self.mean_intrinsic_reward = np.mean(intrinsic_reward)
            # endregion

            # region --- Overwrite Original Rewards ---
            # If recurrent iterate through each sequence in the intrinsic reward array, then iterate through each reward
            # and add it to the original reward of that sample.
            for seq_idx, reward_sequence in enumerate(intrinsic_reward):
                for idx, rew in enumerate(reward_sequence):
                    replay_batch[seq_idx][idx]["reward"] = \
                        replay_batch[seq_idx][idx]["reward"] * (1 - self.reward_scaling_factor) + rew
            # endregion
            return replay_batch
        # endregion

        # region --- NON-RECURRENT ---
        else:
            # region --- Batch Reshaping ---
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes,
                                                               self.action_shape)
            if np.any(np.isnan(action_batch)):
                return replay_batch
            # endregion

            if self.normalize_observations:
                for state in next_state_batch:
                    state -= self.observation_mean
                    state /= self.observation_std
                    state = np.clip(state, -5, 5)

            # region --- Feature Calculation and Learning Step ---
            with tf.device(self.device):
                # Calculate the features for the current next state batch with the target and the prediction model.
                # Then calculate the Mean Squared Error between them.
                with tf.GradientTape() as tape:
                    target_features = self.target_model(next_state_batch)
                    prediction_features = self.prediction_model(next_state_batch)
                    self.loss = self.mse(target_features, prediction_features)
                # Calculate Gradients and apply the weight updates to the prediction model.
                grad = tape.gradient(self.loss, self.prediction_model.trainable_weights)
                self.optimizer.apply_gradients(zip(grad, self.prediction_model.trainable_weights))
            # endregion

            # region --- Intrinsic Reward Calculation ---
            # The intrinsic reward is the L2 error between target and prediction features summed over all features.
            # This results in a 1D-array of rewards if non-recurrent or a 2D-array of rewards if recurrent.
            intrinsic_reward = tf.math.sqrt(
                tf.math.reduce_sum(tf.math.square(target_features - prediction_features), axis=-1)).numpy()

            # Calculate the standard deviation of the intrinsic rewards to normalize them.
            for reward in intrinsic_reward:
                self.reward_deque.append(reward)
            self.reward_std = np.std(self.reward_deque)

            # Normalize the intrinsic reward by the standard deviation
            intrinsic_reward /= self.reward_std
            # Scale the reward
            intrinsic_reward *= self.reward_scaling_factor
            # endregion

            # region --- Store Max and Mean intrinsic rewards
            self.max_intrinsic_reward = np.max(intrinsic_reward)
            self.mean_intrinsic_reward = np.mean(intrinsic_reward)
            # endregion

            # region --- Overwrite Original Rewards ---
            # Iterate through each reward and add it to the original sample.
            # Else just iterate through each reward and add it to the original sample.
            for idx, rew in enumerate(intrinsic_reward):
                replay_batch[idx]["reward"] = \
                    replay_batch[idx]["reward"]*(1-self.reward_scaling_factor)+rew
            # endregion
            return replay_batch
        # endregion

    @staticmethod
    def get_config():
        config_dict = RandomNetworkDistillation.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps):
        if self.index or not len(decision_steps):
            return None
        for idx, state in enumerate(decision_steps.obs):
            state = np.mean(state)
            self.observation_deque.append(state)
        self.observation_mean = np.mean(self.observation_deque)
        self.observation_std = np.std(self.observation_deque)

    def get_logs(self):
        if self.index == 0:
            return {"Exploration/RNDLoss": self.loss,
                    "Exploration/MaxIntrinsicReward": self.max_intrinsic_reward,
                    "Exploration/MeanIntrinsicReward": self.mean_intrinsic_reward}
        else:
            return {}

    def prevent_checkpoint(self):
        return False
