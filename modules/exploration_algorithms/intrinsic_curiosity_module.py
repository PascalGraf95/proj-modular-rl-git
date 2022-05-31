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


class IntrinsicCuriosityModule(ExplorationAlgorithm):
    """
    Basic implementation of the Intrinsic Curiosity Module (ICM)
    """
    Name = "IntrinsicCuriosityModule"
    ActionAltering = False
    IntrinsicReward = True

    ParameterSpace = {
        "FeatureSpaceSize": int,
        "CuriosityScalingFactor": float,
        "ForwardLossWeight": float,
        "LearningRate": float,
    }

    def __init__(self, action_shape, observation_shapes, action_space, parameters, idx):
        self.action_space = action_space
        self.action_shape = action_shape
        self.observation_shapes = observation_shapes

        self.inverse_loss = 0
        self.forward_loss = 0
        self.max_intrinsic_reward = 0
        self.loss = 0
        self.index = idx

        if self.index == 0:
            self.feature_space_size = parameters["FeatureSpaceSize"]
            self.reward_scaling_factor = parameters["CuriosityScalingFactor"]*parameters["ExplorationDegree"]
            self.forward_loss_weight = parameters["ForwardLossWeight"]
            self.mse = MeanSquaredError()
            self.optimizer = Adam(parameters["LearningRate"])
            self.forward_model, self.inverse_model, self.feature_extractor = self.build_network()

    def build_network(self):
        # Feature Extractor: Extracts relevant state features
        if len(self.observation_shapes) == 1:
            feature_input = Input(self.observation_shapes[0])
            x = feature_input
        else:
            feature_input = []
            for obs_shape in self.observation_shapes:
                feature_input.append(Input(obs_shape))
            x = Concatenate()(feature_input)
        x = Dense(self.feature_space_size*4, activation="elu")(x)
        x = Dense(self.feature_space_size*2, activation="elu")(x)
        x = Dense(self.feature_space_size, activation="elu")(x)
        feature_extractor = Model(feature_input, x, name="ICM Feature Extractor")

        # Inverse Model: Predicts the action given features for the current and the next state
        state_feature_input = Input(self.feature_space_size)
        next_state_feature_input = Input(self.feature_space_size)
        x = Concatenate(axis=-1)([state_feature_input, next_state_feature_input])
        x = Dense(32, 'elu')(x)
        x = Dense(self.action_shape, 'tanh')(x)
        inverse_model = Model([state_feature_input, next_state_feature_input], x, name="ICM Inverse Model")

        # Forward Model: Predicts the next state's features given the current state features and the action
        state_feature_input = Input(self.feature_space_size)
        action_input = Input(self.action_shape)
        x = Concatenate(axis=-1)([state_feature_input, action_input])
        x = Dense(32, 'elu')(x)
        x = Dense(self.feature_space_size, 'elu')(x)
        forward_model = Model([state_feature_input, action_input], x, name="ICM Forward Model")

        # Summaries
        forward_model.summary()
        inverse_model.summary()
        feature_extractor.summary()

        return forward_model, inverse_model, feature_extractor

    def learning_step(self, replay_batch):
        if self.index == 0:
            state_batch, action_batch, \
             reward_batch, next_state_batch, \
             done_batch = Learner.get_training_batch_from_replay_batch(replay_batch,
                                                                       self.observation_shapes,
                                                                       self.action_shape)

            # Calculate Loss
            with tf.GradientTape() as tape:
                # Feature Extraction
                state_features = self.feature_extractor(state_batch)
                next_state_features = self.feature_extractor(next_state_batch)

                # Forward Loss
                next_state_prediction = self.forward_model([state_features, action_batch])
                self.forward_loss = self.mse(next_state_features, next_state_prediction)

                # Inverse Loss
                action_prediction = self.inverse_model([state_features, next_state_features])
                self.inverse_loss = self.mse(action_batch, action_prediction)

                # Combined Loss
                self.loss = self.forward_loss*self.forward_loss_weight + self.inverse_loss*(1-self.forward_loss_weight)

            # Calculate Gradients
            grad = tape.gradient(self.loss, [self.forward_model.trainable_weights,
                                             self.inverse_model.trainable_weights,
                                             self.feature_extractor.trainable_weights])
            # Apply Gradients to all models
            self.optimizer.apply_gradients(zip(grad[0], self.forward_model.trainable_weights))
            self.optimizer.apply_gradients(zip(grad[1], self.inverse_model.trainable_weights))
            self.optimizer.apply_gradients(zip(grad[2], self.feature_extractor.trainable_weights))
        return

    def get_intrinsic_reward(self, replay_batch):
        state_batch, action_batch, \
            reward_batch, next_state_batch, \
            done_batch = Learner.get_training_batch_from_replay_batch(replay_batch,
                                                                      self.observation_shapes,
                                                                      self.action_shape)

        # Extract features from the current and next state
        state_features = self.feature_extractor(state_batch)
        next_state_features = self.feature_extractor(next_state_batch)
        # Predict next state features given current features and action
        next_state_prediction = self.forward_model([state_features, action_batch])
        # Calculate the intrinsic reward by MSE between prediction and actual next state feature
        intrinsic_reward = tf.math.reduce_sum(
            tf.math.square(next_state_features - next_state_prediction), axis=-1).numpy()
        # Scale the reward
        intrinsic_reward *= self.reward_scaling_factor
        self.max_intrinsic_reward = np.max(intrinsic_reward)

        # Replace the original reward in the replay batch
        for idx, int_rew in enumerate(intrinsic_reward):
            replay_batch[idx]["reward"] = replay_batch[idx]["reward"]*(1-self.reward_scaling_factor)+int_rew
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = IntrinsicCuriosityModule.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps):
        return

    def get_logs(self):
        if self.index == 0:
            return {"Exploration/InverseLoss": self.inverse_loss,
                    "Exploration/ForwardLoss": self.forward_loss,
                    "Exploration/MaxIntrinsicReward": self.max_intrinsic_reward}
        else:
            return {}

    def prevent_checkpoint(self):
        return False
