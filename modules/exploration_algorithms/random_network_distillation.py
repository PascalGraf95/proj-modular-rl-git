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
        self.reward_scaling_factor = exploration_parameters["CuriosityScalingFactor"] * \
                                     exploration_parameters["ExplorationDegree"]
        self.mse = MeanSquaredError()
        self.optimizer = Adam(exploration_parameters["LearningRate"])

        self.max_intrinsic_reward = 0
        self.loss = 0

        self.prediction_model, self.target_model = self.build_network()

    def build_network(self):
        # Prediction Model: tries to mimic the target model
        feature_input = Input((self.sequence_length, *self.observation_shapes[0]))
        x = Dense(64, activation="elu")(feature_input)
        x = Dense(64, activation="elu")(x)
        x = Dense(self.feature_space_size, activation="linear")(x)
        prediction_model = Model(feature_input, x, name="RND Prediction Model")

        # Target Model: Randomly Initialized
        feature_input = Input((self.sequence_length, *self.observation_shapes[0]))
        x = Dense(64, activation="elu")(feature_input)
        x = Dense(64, activation="elu")(x)
        x = Dense(self.feature_space_size, activation="linear")(x)
        target_model = Model(feature_input, x, name="RND Target Model")

        return prediction_model, target_model

    def learning_step(self, replay_batch):
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)

        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        # Calculate Loss
        with tf.GradientTape() as tape:
            # Feature Prediction
            target_features = self.target_model(state_batch)
            prediction_features = self.prediction_model(state_batch)
            # Loss Calculation
            self.loss = self.mse(target_features, prediction_features)

        # Calculate Gradients and apply the weight updates
        grad = tape.gradient(self.loss, self.prediction_model.trainable_weights)
        self.optimizer.apply_gradients(zip(grad, self.prediction_model.trainable_weights))
        return

    def get_intrinsic_reward(self, replay_batch):
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)

        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        print(state_batch[0].shape, self.target_model.input_shape, len(state_batch))
        # Predict features with the target and prediction model
        target_features = self.target_model(state_batch)
        prediction_features = self.prediction_model(state_batch)
        print("TARGET FEATURES", target_features.shape)
        time.sleep(10)
        # Calculate the intrinsic reward by MSE between prediction and actual next state feature
        intrinsic_reward = tf.math.reduce_sum(
            tf.math.abs(target_features - prediction_features), axis=-1).numpy()
        # Scale the reward
        intrinsic_reward *= self.reward_scaling_factor
        self.max_intrinsic_reward = np.max(intrinsic_reward)

        # Replace the original reward in the replay batch
        for idx, int_rew in enumerate(intrinsic_reward):
            replay_batch[idx]["reward"] = replay_batch[idx]["reward"]*(1-self.reward_scaling_factor)+int_rew
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = RandomNetworkDistillation.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps):
        return

    def get_logs(self, idx):
        return {"Exploration/RNDLoss": self.loss,
                "Exploration/MaxIntrinsicReward": self.max_intrinsic_reward}

    def prevent_checkpoint(self):
        return False
