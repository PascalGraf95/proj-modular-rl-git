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
    # TODO: Optimierungsidee - Aufruf der learning funktion mit dem actor für den der höchste durchschnittliche extrinsische reward erspielt wurde
    def __init__(self, action_shape, observation_shapes,
                 action_space,
                 exploration_parameters,
                 training_parameters, idx):
        self.action_space = action_space
        self.action_shape = action_shape
        self.observation_shapes = observation_shapes

        # Override observation shapes (placeholders for intrinsic reward and exploration policy as inputs)
        modified_observation_shapes = []
        for obs_shape in self.observation_shapes:
            modified_observation_shapes.append(obs_shape)
        modified_observation_shapes.append((1,))
        modified_observation_shapes.append((1,))
        self.observation_shapes = modified_observation_shapes

        self.index = idx
        self.device = '/cpu:0'
        self.recurrent = training_parameters["Recurrent"]
        self.sequence_length = training_parameters["SequenceLength"]
        self.feature_space_size = exploration_parameters["FeatureSpaceSize"]

        # Scale rewards by exploration degree
        self.reward_scaling_factor = exploration_parameters["ExplorationDegree"]["beta"]

        # Categorical Cross-Entropy for discrete action spaces
        # Mean Squared Error for continuous action spaces
        if self.action_space == "DISCRETE":
            self.cce = CategoricalCrossentropy()
        elif self.action_space == "CONTINUOUS":
            self.mse = MeanSquaredError()

        self.optimizer = Adam(exploration_parameters["LearningRate"])
        self.loss = 0
        self.episodic_intrinsic_reward = 0
        self.intrinsic_reward = 0

        # Episodic memory and kernel hyperparameters
        self.k = exploration_parameters["kNearest"]
        self.cluster_distance = exploration_parameters["ClusterDistance"]
        self.eps = exploration_parameters["KernelEpsilon"]
        self.c = exploration_parameters["KernelConstant"]
        self.similarity_max = exploration_parameters["MaximumSimilarity"]
        self.episodic_memory = deque(maxlen=exploration_parameters["EpisodicMemoryCapacity"])
        self.mean_distances = deque(maxlen=self.episodic_memory.maxlen)

        self.feature_extractor, self.embedding_classifier = self.build_network()

    def build_network(self):
        with tf.device(self.device):
            # region Feature Extractor
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
            feature_extractor = Model(feature_input, x, name="ENM Feature Extractor")
            # endregion

            # region Classifier
            if self.recurrent:
                # Add additional time dimension if networks work with recurrent replay batches
                current_state_features = Input((None, *(self.feature_space_size,)))
                next_state_features = Input((None, *(self.feature_space_size,)))
            else:
                current_state_features = Input(self.feature_space_size)
                next_state_features = Input(self.feature_space_size)
            x = Concatenate(axis=-1)([current_state_features, next_state_features])
            x = Dense(128, 'relu')(x)
            if self.action_space == "DISCRETE":
                x = Dense(self.action_shape[0], 'softmax')(x)
            elif self.action_space == "CONTINUOUS":
                x = Dense(self.action_shape, 'tanh')(x)
            embedding_classifier = Model([current_state_features, next_state_features], x, name="ENM Classifier")
            # endregion

            # region Model compilation and plotting
            if self.action_space == "DISCRETE":
                embedding_classifier.compile(loss=self.cce, optimizer=self.optimizer)
            elif self.action_space == "CONTINUOUS":
                embedding_classifier.compile(loss=self.mse, optimizer=self.optimizer)

            # Model plots
            try:
                plot_model(feature_extractor, "plots/ENM_FeatureExtractor.png", show_shapes=True)
                plot_model(embedding_classifier, "plots/ENM_EmbeddingClassifier.png", show_shapes=True)
            except ImportError:
                print("Could not create model plots for ENM.")

            # Summaries
            feature_extractor.summary()
            embedding_classifier.summary()
            # endregion

            return feature_extractor, embedding_classifier

            '''# List with two dictionaries in it, one for each network.
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

            return feature_extractor, embedding_classifier'''

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
            # Calculate Loss
            with tf.GradientTape() as tape:
                # Calculate features of current and next state
                state_features = self.feature_extractor(state_batch)
                next_state_features = self.feature_extractor(next_state_batch)

                # Predict actions based on extracted features
                action_prediction = self.embedding_classifier([state_features, next_state_features])

                # Calculate inverse loss
                if self.action_space == "DISCRETE":
                    # TODO: Turn into real and working code
                    # Encode true action as one hot vector encoding
                    num_actions = self.action_shape[:]
                    true_actions_one_hot = tf.one_hot(action_batch, num_actions).numpy()

                    # Compute Loss via Categorical Cross Entropy
                    self.loss = self.cce(true_actions_one_hot, action_prediction)

                elif self.action_space == "CONTINUOUS":
                    # Compute Loss via Mean Squared Error
                    self.loss = self.mse(action_batch, action_prediction)

            # Calculate Gradients
            grad = tape.gradient(self.loss, [self.embedding_classifier.trainable_weights,
                                             self.feature_extractor.trainable_weights])
            # Apply Gradients to all models
            self.optimizer.apply_gradients(zip(grad[0], self.embedding_classifier.trainable_weights))
            self.optimizer.apply_gradients(zip(grad[1], self.feature_extractor.trainable_weights))
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
        if not len(decision_steps.obs[0]):
            self.mean_distances.append(0)
            return 0

        # Extract relevant features from current state
        if self.recurrent:
            # Add additional time dimension if recurrent networks are used
            current_state = [tf.expand_dims(state, axis=1) for state in decision_steps.obs]
            # Calculate state embedding and get rid of additional time dimension through index 0
            state_embedding = self.feature_extractor(current_state)[0]
        else:
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

        # Calculate similarity (will increase as agent collects more and more states similar to each other)
        K = self.eps / (topk_emb_distances + self.eps)
        similarity = np.sqrt(np.sum(K)) + self.c

        # Check for similarity boundaries and return intrinsic episodic reward
        if np.isnan(similarity) or (similarity > self.similarity_max):
            self.episodic_intrinsic_reward = 0
        else:
            # 1/similarity to encourage visiting states with lower similarity
            self.episodic_intrinsic_reward = self.reward_scaling_factor * (1 / similarity)

        return self.episodic_intrinsic_reward

    def get_logs(self):
        return {"Exploration/EpisodicLoss": self.loss,
                "Exploration/Reward_Act{:02d}_{:.4f}".format(self.index, self.reward_scaling_factor): self.episodic_intrinsic_reward}

    def reset(self):
        """Empty episodic memory and clear euclidean distance metrics."""
        #self.mean_distances.clear()
        #self.episodic_memory.clear()
        return

    def prevent_checkpoint(self):
        return False
