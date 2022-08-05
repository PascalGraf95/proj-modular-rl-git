import random

import numpy as np
from ..misc.replay_buffer import FIFOBuffer
from .exploration_algorithm_blueprint import ExplorationAlgorithm
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, BinaryCrossentropy
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
from ..misc.utility import modify_observation_shapes


class EpisodicCuriosity(ExplorationAlgorithm):
    """
    Basic implementation of Episodic Curiosity Through Reachability (ECR)

    https://arxiv.org/abs/1810.02274
    """
    Name = "EpisodicCuriosity"
    ActionAltering = False
    IntrinsicReward = True

    ParameterSpace = {
        "FeatureSpaceSize": int,
        "LearningRate": float,
        "EpisodicMemoryCapacity": int
    }

    def __init__(self, action_shape, observation_shapes,
                 action_space,
                 exploration_parameters,
                 training_parameters, idx):
        self.action_space = action_space
        self.action_shape = action_shape
        self.observation_shapes = observation_shapes
        self.observation_shapes_modified = observation_shapes

        self.index = idx
        self.device = '/cpu:0'

        # Modify observation shapes for sampling later on
        self.observation_shapes_modified = modify_observation_shapes(self.observation_shapes, self.action_shape)
        self.num_additional_obs_values = len(self.observation_shapes_modified) - len(self.observation_shapes)

        # Parameters required during network build-up
        self.episodic_curiosity_built = False
        self.recurrent = training_parameters["Recurrent"]
        self.sequence_length = training_parameters["SequenceLength"]
        self.feature_space_size = exploration_parameters["FeatureSpaceSize"]

        # Loss metrics
        self.cce = CategoricalCrossentropy()
        self.bce = BinaryCrossentropy()
        self.optimizer = Adam(exploration_parameters["LearningRate"])
        self.loss = 0
        self.intrinsic_reward = 0

        # region Episodic Curiosity module parameters
        self.k = 5
        self.novelty_threshold = 0
        self.beta = 1
        self.alpha = 1
        self.gamma = 2
        # TODO: Currently episodic memory as ring-buffer, make overflow mechanic random?
        self.episodic_memory = deque(maxlen=exploration_parameters["EpisodicMemoryCapacity"])
        self.reset_episodic_memory = exploration_parameters["ResetEpisodicMemory"]

        self.feature_extractor, self.comparator_network = self.build_network()
        # endregion

    def build_network(self):
        with tf.device(self.device):
            # region Embedding Network
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
            x = Dense(32, activation="relu")(x)
            x = Dense(self.feature_space_size, activation="relu")(x)
            feature_extractor = Model(feature_input, x, name="ECR Feature Extractor")
            # endregion

            # region Comparator
            if self.recurrent:
                # Add additional time dimension if networks work with recurrent replay batches
                current_state_features = Input((None, *(self.feature_space_size,)))
                next_state_features = Input((None, *(self.feature_space_size,)))
            else:
                current_state_features = Input(self.feature_space_size)
                next_state_features = Input(self.feature_space_size)
            x = Concatenate(axis=-1)([current_state_features, next_state_features])
            x = Dense(32, activation="relu")(x)
            x = Dense(32, activation="relu")(x)
            x = Dense(1, activation='sigmoid')(x)
            comparator_network = Model([current_state_features, next_state_features], x, name="ECR Comparator")
            # endregion

            # region Model compilation and plotting
            comparator_network.compile(loss=self.bce, optimizer=self.optimizer)

            # Model plots
            try:
                plot_model(feature_extractor, "plots/ECR_FeatureExtractor.png", show_shapes=True)
                plot_model(comparator_network, "plots/ECR_Comparator.png", show_shapes=True)
            except ImportError:
                print("Could not create model plots for ECR.")

            # Summaries
            feature_extractor.summary()
            comparator_network.summary()
            # endregion

            return feature_extractor, comparator_network
            # endregion

    def learning_step(self, replay_batch):
        # region --- Batch Reshaping ---
        if self.recurrent:
            state_batch, action_batch, _, _, _ \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes_modified,
                                                                         self.action_shape, self.sequence_length)
        else:
            return "Exploration algorithm 'ECR' does not work with non-recurrent agents. Learning step NOT executed."

        if np.any(np.isnan(action_batch)):
            return replay_batch
        # endregion

        # region --- Learning Step ---
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                # Clear additional observation parts added during acting as they must not be used by the exploration algorithms
                state_batch = state_batch[:-self.num_additional_obs_values]

                # Calculate features
                state_features = self.feature_extractor(state_batch)

                # Create training data (unique feature-pairs with respective reachability information as labels)
                x1_batch, x2_batch, y_true_batch = [], [], []
                for sequence in state_features:
                    x1, x2, y_true = self.get_training_data(sequence)
                    x1_batch.append(x1)
                    x2_batch.append(x2)
                    y_true_batch.append(y_true)

                # Cast arrays for comparator to output correct shape
                x1_batch = np.array(x1_batch)
                x2_batch = np.array(x2_batch)

                # Calculate reachability between observation pairs
                y_pred = self.comparator_network([x1_batch, x2_batch])

                # Calculate Binary Cross-Entropy Loss
                self.loss = self.bce(y_true_batch, y_pred)

            # Calculate Gradients
            grad = tape.gradient(self.loss, self.comparator_network.trainable_weights)

            # Apply Gradients to all models
            self.optimizer.apply_gradients(zip(grad, self.comparator_network.trainable_weights))

        # endregion
        return

    def get_training_data(self, sequence):
        """
        Create training data through forming of random observation pairs and calculating whether the elements of those
        pairs are reachable from one to each other within k-steps. Allocation process is done randomly and differs from
        the original paper where a sliding window based approach is used.

        Parameters
        ----------
        sequence:
            Contains the observation values of a sequence from the state_batch.

        Returns
        -------
        x1:
            First elements of the observation index pairs.
        x2:
            Second elements of the observation index pairs.
        labels: int
            Reachability between x1 and x2 elements and therefore the ground truth of the training data. (0 == not
            reachable within k-steps, 1 == reachable within k-steps)
        """
        # Create Index Array
        sequence_len = len(sequence)
        sequence_indices = np.arange(sequence_len)

        # Shuffle sequence indices randomly
        np.random.shuffle(sequence_indices)
        # Get middle index
        sequence_middle = sequence_len // 2
        # Divide Index-Array into two equally sized parts (Right half gets cutoff if sequence length is odd)
        sequence_indices_left, sequence_indices_right = sequence_indices[:sequence_middle], \
                                                        sequence_indices[sequence_middle:2 * sequence_middle]
        idx_differences = np.abs(sequence_indices_left - sequence_indices_right)
        x1 = sequence.numpy()[sequence_indices_left]
        x2 = sequence.numpy()[sequence_indices_right]

        # States are reachable (== 1) one from each other if step-difference between them is smaller than k
        labels = np.where(idx_differences < self.k, 1, 0)
        labels = tf.expand_dims(labels, axis=-1)  # necessary as comparator output is 3D

        return x1, x2, labels

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    def act(self, decision_steps, terminal_steps):
        if not len(decision_steps.obs[0]):
            current_state = terminal_steps.obs
        else:
            current_state = decision_steps.obs

        # Extract relevant features from current state
        if self.recurrent:
            # Add additional time dimension if recurrent networks are used
            current_state = [tf.expand_dims(state, axis=1) for state in current_state]
            # '[0]' -> to get rid of time dimension directly after network inference
            state_embedding = self.feature_extractor(current_state)[0]
        else:
            print("Exploration algorithm 'ECR' does not work with non-recurrent agents. Acting step NOT executed.")
            return 0

        # First observation must be added to episodic memory before executing further calculations
        if not self.episodic_memory.__len__():
            self.episodic_memory.append(state_embedding)
            return 0

        # Create array with length of the current episodic memory containing same copies of the current state embedding
        state_embedding_array = np.empty([self.episodic_memory.__len__(), state_embedding.shape[0],
                                          state_embedding.shape[1]])
        state_embedding_array[:] = state_embedding

        # Get reachability buffer
        reachability_buffer = self.comparator_network([state_embedding_array, np.array(self.episodic_memory)])

        # Aggregate the content of the reachability buffer to calculate similarity-score of current embedding
        similarity_score = np.percentile(reachability_buffer, 90)
        self.intrinsic_reward = self.alpha * (self.beta - similarity_score)

        # Add state to episodic memory if similarity score is large enough
        if self.intrinsic_reward > self.novelty_threshold:
            self.episodic_memory.append(state_embedding)

        return self.intrinsic_reward

    def get_logs(self):
        if self.index == 0:
            return {"Exploration/EpisodicLoss": self.loss,
                    "Exploration/IntrinsicReward": self.intrinsic_reward}
        else:
            return {"Exploration/EpisodicLoss": self.loss}

    def reset(self):
        """Empty episodic memory."""
        if self.reset_episodic_memory:
            self.episodic_memory.clear()
        return

    @staticmethod
    def get_config():
        config_dict = EpisodicCuriosity.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def prevent_checkpoint(self):
        return False
