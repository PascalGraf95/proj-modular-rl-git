import numpy as np
from ..misc.replay_buffer import FIFOBuffer
from .exploration_algorithm_blueprint import ExplorationAlgorithm
from tensorflow import keras
from keras.losses import MeanSquaredError
from keras.optimizers import Adam, SGD
from modules.misc.network_constructor import construct_network
import tensorflow as tf
from modules.training_algorithms.agent_blueprint import Learner
import itertools
from keras import Input, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
from modules.misc.utility import modify_observation_shapes
import time
from collections import deque
from keras.utils import plot_model


class RandomNetworkDistillation(ExplorationAlgorithm):
    """
      Implementation of Random Network Distillation (RND) that works based on the step-based reward
      calculation principle of Agent57.
      """
    Name = "RandomNetworkDistillationAlter"
    ActionAltering = False
    IntrinsicReward = True

    ParameterSpace = {
        "FeatureSpaceSize": int,
        "LearningRate": float,
    }

    def __init__(self, action_shape, observation_shapes,
                 action_space,
                 parameters,
                 trainer_configuration, idx):
        # region - Action and Observation Spaces -
        self.action_space = action_space
        self.action_shape = action_shape
        # The observation shapes have to be known for network construction. Furthermore, they might be modified
        # by the augmentation of other metrics via feedback.
        self.observation_shapes = observation_shapes
        self.observation_shapes_modified = self.observation_shapes
        # endregion

        # region - Misc -
        self.index = idx
        self.device = '/cpu:0'
        self.training_step = 0
        self.loss = 0
        self.intrinsic_reward = 0
        # endregion

        # region - Network Parameters -
        # Recurrent parameters determine the dimensionality of replay batches.
        self.recurrent = trainer_configuration["Recurrent"]
        self.sequence_length = trainer_configuration["SequenceLength"]
        # The chosen Feature Space Size corresponds to the number of output neurons in the feature extractor and thus
        # determines how much the original state is compressed.
        self.feature_space_size = parameters["FeatureSpaceSize"]
        self.optimizer = Adam(parameters["LearningRate"])
        self.mse = MeanSquaredError()
        # endregion

        # region - Random Network Distillation specific Parameters -
        self.normalize_observations = parameters["ObservationNormalization"]
        self.observation_deque = deque(maxlen=1000)
        self.observation_mean = 0
        self.observation_std = 1
        self.reward_deque = deque(maxlen=1000)
        self.reward_mean = 0
        self.reward_std = 1
        # endregion

        # region - Modify observation shapes and construct Neural Networks -
        # Takes into account that prior actions, rewards and policies might be used for state augmentation.
        # However, their values should not be considered for RND. The modified observation shape is only used for
        # sample batch preprocessing.
        self.observation_shapes_modified = modify_observation_shapes(self.observation_shapes, self.action_shape,
                                                                     self.action_space,
                                                                     trainer_configuration["ActionFeedback"],
                                                                     trainer_configuration["RewardFeedback"],
                                                                     trainer_configuration["PolicyFeedback"])
        self.num_additional_obs_values = len(self.observation_shapes_modified) - len(self.observation_shapes)
        # RND utilizes two network architectures. A prediction model is thereby responsible for predicting the output
        # values of a non-changing target network.
        self.prediction_model, self.target_model = self.build_network()
        # endregion

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
            target_model = construct_network(network_parameters[1], plot_network_model=True)

            try:
                plot_model(prediction_model, "plots/RND_Prediction_Model.png", show_shapes=True)
                plot_model(target_model, "plots/RND_Target_Model.png", show_shapes=True)
            except ImportError:
                print("Could not create model plots for RND.")

            return prediction_model, target_model

    def learning_step(self, replay_batch):
        # region - Batch Reshaping -
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes_modified,
                                                                         self.action_shape, self.sequence_length)

        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes,
                                                               self.action_shape)

        if np.any(np.isnan(action_batch)):
            return replay_batch

        # Clear augmented state parts such as rewards, actions and policy indices as they must not be used by
        # the exploration algorithms, i.e. closeness of states should not depend on the intrinsic reward or the
        # exploration strategy followed at that time. Furthermore, providing the taken action would make the problem
        # trivial.
        if self.num_additional_obs_values:
            next_state_batch = next_state_batch[:-self.num_additional_obs_values]
        # endregion

        # region - Prediction Model training -
        # The Prediction Model is trained on the mse between the predicted values of the Target and Prediction Model.
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                # Calculate features
                target_features = self.target_model(next_state_batch)
                prediction_features = self.prediction_model(next_state_batch)

                # Compute Loss via Mean Squared Error
                self.loss = self.mse(target_features, prediction_features)

            # Calculate Gradients
            grad = tape.gradient(self.loss, self.prediction_model.trainable_weights)
            # Apply Gradients to the Prediction Model
            self.optimizer.apply_gradients(zip(grad, self.prediction_model.trainable_weights))
        # endregion
        return

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = RandomNetworkDistillation.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def epsilon_greedy(self, decision_steps):
        return None

    def act(self, decision_steps, terminal_steps):
        if not len(decision_steps.obs[0]):
            current_state = terminal_steps.obs
        else:
            current_state = decision_steps.obs

        # region - Observation normalization -
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
        # endregion

        # region - Feature calculation -
        if self.recurrent:
            # Add additional time dimension if recurrent networks are used
            current_state = [tf.expand_dims(state, axis=1) for state in current_state]
            # Calculate state features
            target_features = self.target_model(current_state)[0]
            prediction_features = self.prediction_model(current_state)[0]
        else:
            # Calculate state features
            target_features = self.target_model(current_state)
            prediction_features = self.prediction_model(current_state)
        # endregion

        # region - Reward calculation and normalization -
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
        # endregion

        return self.intrinsic_reward

    def get_logs(self):
        if self.index == 0:
            return {"Exploration/LifeLongLoss": self.loss,
                    "Exploration/IntrinsicReward": self.intrinsic_reward}
        else:
            return {"Exploration/LifeLongLoss": self.loss}

    def prevent_checkpoint(self):
        return False
