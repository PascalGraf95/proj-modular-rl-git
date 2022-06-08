import numpy as np
from ..misc.replay_buffer import FIFOBuffer
from .exploration_algorithm_blueprint import ExplorationAlgorithm
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
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


class EpisodicNoveltyModule(ExplorationAlgorithm):
    """
    Basic implementation of Episodic Novelty Module (ENM)
    The computation of intrinsic episodic rewards is done for each actor and after every environment step (see act()).
    Logic can be compared to pseudo code within the respective paper:
    https://openreview.net/pdf?id=Sye57xStvB
    """
    Name = "EpisodicNoveltyModule"
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
        self.index = idx
        self.device = '/cpu:0'

        self.recurrent = training_parameters["Recurrent"]
        self.sequence_length = training_parameters["SequenceLength"]

        self.feature_space_size = exploration_parameters["FeatureSpaceSize"]
        self.beta = exploration_parameters["ExplorationDegree"]["beta"]

        # Categorical Cross-Entropy for discrete action spaces
        # Mean Squared Error for continuous action spaces
        if self.action_space == "DISCRETE":
            self.cce = CategoricalCrossentropy()
        elif self.action_space == "CONTINUOUS":
            self.mse = MeanSquaredError()

        self.optimizer = Adam(exploration_parameters["LearningRate"])
        self.loss = 0

        # Episodic memory and kernel hyperparameters
        # TODO: Transfer into trainer_config.yaml
        self.k = 10  # exploration_parameters["kNearest"]
        self.cluster_distance = 0.008  # exploration_parameters["ClusterDistance"]
        self.eps = 0.0001  # exploration_parameters["KernelEpsilon"]
        self.c = 0.001  # exploration_parameters["KernelConstant"]
        self.similarity_max = 8  # exploration_parameters["MaximumSimilarity"]
        self.episodic_memory = deque(maxlen=1000)  # deque(maxlen=exploration_parameters["EpisodicMemoryCapacity"])
        self.mean_distances = deque(maxlen=self.episodic_memory.maxlen)
        self.episodic_intrinsic_reward = 0
        self._FIRST_ITER = True

        print('******************************************************************')
        print('Actor Index:', self.index)
        print('Exploration Degree:', exploration_parameters["ExplorationDegree"])
        print('******************************************************************')

        self.feature_extractor, self.embedding_classifier = self.build_network()

    def build_network(self):
        with tf.device(self.device):
            # List with two dictionaries in it, one for each network.
            network_parameters = [{}, {}]
            # region --- Feature Extraction Model ---
            # This model outputs an embedding vector consisting of observation components the agent can influence
            # through its actions.
            # - Network Name -
            network_parameters[0]['NetworkName'] = "ENM_FeatureExtractor"
            # - Network Architecture-
            network_parameters[0]['VectorNetworkArchitecture'] = "Dense"
            network_parameters[0]['VisualNetworkArchitecture'] = "CNN"
            network_parameters[0]['Filters'] = 32
            network_parameters[0]['Units'] = 32
            network_parameters[0]['TargetNetwork'] = False
            # - Input / Output / Initialization -
            network_parameters[0]['Input'] = self.observation_shapes
            network_parameters[0]['Output'] = [self.feature_space_size]
            #network_parameters[0]['OutputActivation'] = [None]
            network_parameters[0]['OutputActivation'] = ["relu"]
            # - Recurrent Parameters -
            network_parameters[0]['Recurrent'] = False

            # region --- Embedding Classifier Model ---
            # This model tries to predict the action used for the transition between two states.
            # - Network Name -
            network_parameters[1] = network_parameters[0].copy()
            network_parameters[1]['NetworkName'] = "ENM_EmbeddingClassifier"
            # - Network Architecture-
            network_parameters[1]['VectorNetworkArchitecture'] = "SingleDense"
            network_parameters[1]['Units'] = 64
            # - Input / Output / Initialization -
            network_parameters[1]['Input'] = [self.feature_space_size, self.feature_space_size]
            if self.action_space == "DISCRETE":
                network_parameters[1]['Output'] = [self.action_shape[0]]
                network_parameters[1]['OutputActivation'] = ["softmax"]
            elif self.action_space == "CONTINUOUS":
                network_parameters[1]['Output'] = [self.action_shape]
                network_parameters[1]['OutputActivation'] = ["tanh"]

            feature_extractor = construct_network(network_parameters[0], plot_network_model=True)
            embedding_classifier = construct_network(network_parameters[1], plot_network_model=True)

            return feature_extractor, embedding_classifier

    def learning_step(self, replay_batch):
        # region --- RECURRENT
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)

            if np.any(np.isnan(action_batch)):
                return replay_batch

            for state_sequence, next_state_sequence, action_sequence in zip(state_batch[0], next_state_batch[0],
                                                                            action_batch):
                # Cast as lists (MUST BE DONE)
                state_sequence = [state_sequence]
                next_state_sequence = [next_state_sequence]

                with tf.device(self.device):
                    # Calculate Loss
                    with tf.GradientTape() as tape:
                        # Calculate features of this and next state
                        state_features = self.feature_extractor(state_sequence)
                        next_state_features = self.feature_extractor(next_state_sequence)

                        # Inverse Loss
                        action_prediction = self.embedding_classifier([state_features, next_state_features])

                        if self.action_space == "DISCRETE":
                            # TODO: Turn into real and working code
                            # Encode true action as one hot vector encoding
                            num_actions = self.action_shape[:]
                            true_actions_one_hot = tf.one_hot(action_sequence, num_actions).numpy()

                            # Compute Loss via Categorical Cross Entropy
                            self.inverse_loss = self.cce(true_actions_one_hot, action_prediction)

                        elif self.action_space == "CONTINUOUS":
                            # Compute Loss via Mean Squared Error
                            self.inverse_loss = self.mse(action_sequence, action_prediction)

                        self.loss = self.inverse_loss

                    # Calculate Gradients
                    grad = tape.gradient(self.loss, [self.embedding_classifier.trainable_weights,
                                                     self.feature_extractor.trainable_weights])
                    # Apply Gradients to all models
                    self.optimizer.apply_gradients(zip(grad[0], self.embedding_classifier.trainable_weights))
                    self.optimizer.apply_gradients(zip(grad[1], self.feature_extractor.trainable_weights))
        # endregion

        # region --- NON-RECURRENT
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes,
                                                               self.action_shape)

            if np.any(np.isnan(action_batch)):
                return replay_batch
            # endregion

            with tf.device(self.device):
                # Calculate Loss
                with tf.GradientTape() as tape:
                    # Calculate features of this and next state
                    state_features = self.feature_extractor(state_batch)
                    next_state_features = self.feature_extractor(next_state_batch)

                    # Inverse Loss
                    action_prediction = self.embedding_classifier([state_features, next_state_features])

                    if self.action_space == "DISCRETE":
                        # TODO: Turn into real and working code
                        # Encode true action as one hot vector encoding
                        num_actions = self.action_shape[:]
                        true_actions_one_hot = tf.one_hot(action_batch, num_actions).numpy()

                        # Compute Loss via Categorical Cross Entropy
                        self.inverse_loss = self.cce(true_actions_one_hot, action_prediction)

                    elif self.action_space == "CONTINUOUS":
                        # Compute Loss via Mean Squared Error
                        self.inverse_loss = self.mse(action_batch, action_prediction)

                    self.loss = self.inverse_loss

                # Calculate Gradients
                grad = tape.gradient(self.loss, [self.embedding_classifier.trainable_weights,
                                                 self.feature_extractor.trainable_weights])
                # Apply Gradients to all models
                self.optimizer.apply_gradients(zip(grad[0], self.embedding_classifier.trainable_weights))
                self.optimizer.apply_gradients(zip(grad[1], self.feature_extractor.trainable_weights))
        # endregion
        return

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = EpisodicNoveltyModule.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        # TODO: Add variable descriptions
        """Calculate intrinsically-based episodic reward through similarity comparison of current state embedding
        with prior ones present within episodic memory. Needs to be calculated on every actor step and therefore within
        this function and not get_intrinsic_reward().

        Notes:
        - The code within this function is called from play_one_step() within agent_blueprint.py
        - Computed intrinsic reward must be returned to actor and from there sent to replay_buffer
        """
        # Extract relevant features from current state
        state_embedding = self.feature_extractor(decision_steps.obs)

        # Calculate the euclidean distances between current state embedding and the ones within episodic memory (N_k)
        embedding_distances = [np.linalg.norm(mem_state_embedding - state_embedding)
                               for mem_state_embedding in self.episodic_memory]

        # Add state to episodic memory
        self.episodic_memory.append(state_embedding)

        # Get list of top k distances (d_k)
        topk_emb_distances = np.sort(embedding_distances)[:self.k]  # ascending order

        # Calculate mean distance value of current top k distances (d_m)
        if np.any(topk_emb_distances):
            self.mean_distances.append(np.mean(topk_emb_distances))
        else:
            # Mean distance will be zero for first iteration, as episodic memory is empty
            self.mean_distances.append(0)
            return 0

        # Normalize the distances with moving average of mean distance
        topk_emb_distances_normalized = topk_emb_distances / np.mean(self.mean_distances)

        # Cluster the normalized distances
        topk_emb_distances = np.where(topk_emb_distances_normalized - self.cluster_distance > 0,
                                      topk_emb_distances_normalized - self.cluster_distance, 0)

        '''
        if len(self.mean_distances):
            if np.mean(self.mean_distances)>0:
                # Normalize the distances with moving average of mean distance
                topk_emb_distances_normalized = topk_emb_distances / np.mean(self.mean_distances)

                # Cluster the normalized distances
                topk_emb_distances = np.where(topk_emb_distances_normalized - self.cluster_distance > 0,
                                         topk_emb_distances_normalized - self.cluster_distance, 0)
            else:
                return 0
        else:
            return 0'''

        # Calculate similarity (will increase as agent collects more and more states similar to each other)
        K = self.eps / (topk_emb_distances + self.eps)
        similarity = np.sqrt(np.sum(K)) + self.c

        # Check for similarity boundaries and return intrinsic episodic reward
        if np.isnan(similarity) or (similarity > self.similarity_max):
            self.episodic_intrinsic_reward = 0
        else:
            # 1/similarity to encourage visiting states with lower similarity
            self.episodic_intrinsic_reward = self.beta * (1 / similarity)

        # print("Episodic Reward: ", self.episodic_intrinsic_reward)
        return self.episodic_intrinsic_reward


    def get_logs(self):
        return {"Exploration/EpisodicLoss": self.loss,
                "Exploration/EpisodicReward": self.episodic_intrinsic_reward}

    def reset(self):
        """Empty episodic memory and clear euclidean distance metrics."""
        # if self.episodic_memory.__len__() > self.episodic_memory.maxlen:
        # print('*****Episodic Novelty Module reset with length: ', self.episodic_memory.__len__())
        # self.mean_distances.clear()
        # self.episodic_memory.clear()
        return

    def prevent_checkpoint(self):
        return False
