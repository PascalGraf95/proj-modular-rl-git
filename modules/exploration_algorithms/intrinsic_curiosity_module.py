import numpy as np
from ..misc.replay_buffer import ReplayBuffer
from .exploration_algorithm_blueprint import ExplorationAlgorithm
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam
from ..misc.network_constructor import construct_network
import tensorflow as tf
from ..training_algorithms.agent_blueprint import Agent
import itertools


class IntrinsicCuriosityModule(ExplorationAlgorithm):
    Name = "IntrinsicCuriosityModule"
    ActionAltering = False
    IntrinsicReward = True

    ParameterSpace = {
        "FeatureSpaceSize": int,
        "ScalingFactor": float,
        "BatchSize": int,
        "ForwardModelLearningRate": float,
        "InverseModelLearningRate": float,
    }

    def __init__(self, action_shape, observation_shapes, action_space, parameters):
        self.action_space = action_space
        self.action_shape = action_shape
        self.observation_shapes = observation_shapes

        self.feature_space_size = parameters["FeatureSpaceSize"]
        self.scaling_factor = parameters["ScalingFactor"]
        self.batch_size = parameters["BatchSize"]
        self.mse = MeanSquaredError()
        self.forward_optimizer = Adam(parameters["ForwardModelLearningRate"])
        self.inverse_optimizer = Adam(parameters["InverseModelLearningRate"])

        self.inverse_loss = 0
        self.forward_loss = 0
        self.mean_intrinsic_reward = 0

        self.forward_model, self.inverse_model, self.feature_extractor = self.build_network()

    def build_network(self):
        network_parameters = [{}, {}, {}]
        # Forward Model
        network_parameters[0]['Input'] = [self.feature_space_size, self.action_shape]
        network_parameters[0]['Output'] = [self.feature_space_size]
        network_parameters[0]['OutputActivation'] = ["selu"]
        network_parameters[0]['VectorNetworkArchitecture'] = "SingleDense"
        network_parameters[0]['Units'] = 256
        network_parameters[0]['NetworkType'] = "ICMForwardModel"

        # Feature Extractor
        network_parameters[1]['Input'] = self.observation_shapes
        network_parameters[1]['InputResize'] = (42, 42)
        network_parameters[1]['Output'] = [None]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['VisualNetworkArchitecture'] = "ICMCNN"
        network_parameters[1]['VectorNetworkArchitecture'] = "SmallDense"
        network_parameters[1]['Units'] = self.feature_space_size
        network_parameters[1]['Filters'] = 32
        network_parameters[1]['NetworkType'] = "ICMFeatureExtractor"

        # Inverse Model
        network_parameters[2]['Input'] = [self.feature_space_size, self.feature_space_size]
        network_parameters[2]['Output'] = [self.action_shape]
        network_parameters[2]['OutputActivation'] = ["tanh"]
        network_parameters[2]['VectorNetworkArchitecture'] = "SingleDense"
        network_parameters[2]['Units'] = 256
        network_parameters[2]['NetworkType'] = "ICMInverseModel"

        # Build
        forward_model = construct_network(network_parameters[0])
        feature_extractor = construct_network(network_parameters[1])
        inverse_model = construct_network(network_parameters[2])

        return forward_model, inverse_model, feature_extractor

    def update(self, state_batch, action_batch, next_state_batch):
        feature_vector_state = self.feature_extractor(state_batch)
        feature_vector_next_state = self.feature_extractor(next_state_batch)
    
        with tf.GradientTape() as tape:
            forward_model_feature_prediction = self.forward_model([feature_vector_state, action_batch])
            self.forward_loss = self.mse(feature_vector_next_state, forward_model_feature_prediction)
        grad = tape.gradient(self.forward_loss, self.forward_model.trainable_weights)
        self.forward_optimizer.apply_gradients(zip(grad, self.forward_model.trainable_weights))
    
        with tf.GradientTape() as tape:
            feature_vector_state = self.feature_extractor(state_batch)
            feature_vector_next_state = self.feature_extractor(next_state_batch)
            action_prediction = self.inverse_model([feature_vector_state, feature_vector_next_state])
            self.inverse_loss = self.mse(action_batch, action_prediction)

        grad = tape.gradient(self.inverse_loss, [self.inverse_model.trainable_weights,
                                                 self.feature_extractor.trainable_weights])
        self.inverse_optimizer.apply_gradients(zip(grad[0], self.inverse_model.trainable_weights))
        self.inverse_optimizer.apply_gradients(zip(grad[1], self.feature_extractor.trainable_weights))
        return

    @staticmethod
    def get_config():
        config_dict = IntrinsicCuriosityModule.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps):
        return

    def get_logs(self):
        return {"Exploration/InverseLoss": self.inverse_loss,
                "Exploration/ForwardLoss": self.forward_loss,
                "Exploration/MeanIntrinsicReward": self.mean_intrinsic_reward}

    def get_intrinsic_reward(self, replay_batch):
        state_batch, action_batch, \
            reward_batch, next_state_batch, \
            done_batch = Agent.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        feature_vector_state = self.feature_extractor(state_batch)
        feature_vector_next_state = self.feature_extractor(next_state_batch)

        forward_model_feature_prediction = self.forward_model([feature_vector_state, action_batch])
        intrinsic_reward = tf.math.reduce_sum(tf.math.square(feature_vector_next_state - forward_model_feature_prediction), axis=-1).numpy()
        intrinsic_reward *= self.scaling_factor
        self.mean_intrinsic_reward = np.mean(intrinsic_reward)
        # print("Max and Mean Intrinsic Reward: ", np.max(intrinsic_reward), np.mean(intrinsic_reward))
        for idx, intr_rew in enumerate(intrinsic_reward):
            replay_batch[idx]["reward"] += intr_rew
        return replay_batch

    def learning_step(self):
        return

    def prevent_checkpoint(self):
        return False

    def calculate_intrinsic_reward(self, replay_buffer: ReplayBuffer):
        if replay_buffer.new_unmodified_samples < self.batch_size:
            return False

        new_replay_samples = list(itertools.islice(replay_buffer.buffer,
                                                   len(replay_buffer.buffer)-replay_buffer.new_unmodified_samples,
                                                   len(replay_buffer.buffer)))
        state_batch, action_batch, \
            reward_batch, next_state_batch, \
            done_batch = Agent.get_training_batch_from_replay_batch(new_replay_samples, self.observation_shapes, self.action_shape)
        self.update(state_batch, action_batch, next_state_batch)
        return True
