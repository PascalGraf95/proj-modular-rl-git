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

class NeverGiveUp(ExplorationAlgorithm):
    """
    Basic implementation of Never Give Up (NGU) (-> Incorporates ENM and RND)
    The computation of intrinsic episodic rewards is done for each actor and for every environment step.

    ***NOTE about naming***
    - RND is described as the lifelong novelty module of the NGU reward generator
    - ENM is described as the episodic novelty module of the NGU reward generator

    Logic can be compared to pseudo code within the respective paper:
    https://openreview.net/pdf?id=Sye57xStvB
    """
    Name = "NeverGiveUp"
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
        self.episodic_novelty_module_built = False

        self.recurrent = training_parameters["Recurrent"]
        self.sequence_length = training_parameters["SequenceLength"]

        self.feature_space_size = exploration_parameters["FeatureSpaceSize"]
        self.beta = exploration_parameters["ExplorationDegree"]["beta"]

        print('******************************************************************')
        print('Actor Index:', self.index)
        print('Exploration Policy:', exploration_parameters["ExplorationDegree"])
        print('******************************************************************')

        # Categorical Cross-Entropy for discrete action spaces
        # Mean Squared Error for continuous action spaces
        if self.action_space == "DISCRETE":
            self.cce = CategoricalCrossentropy()
        elif self.action_space == "CONTINUOUS":
            self.mse = MeanSquaredError()

        self.optimizer = Adam(exploration_parameters["LearningRate"])
        self.lifelong_loss = 0
        self.episodic_loss = 0
        #self.enm_reward = 0
        #self.rnd_reward = 0  # Refers to alpha, a curiosity scaling factor
        self.intrinsic_reward = 0

        # region Episodic novelty module
        self.k = exploration_parameters["kNearest"]
        self.cluster_distance = exploration_parameters["ClusterDistance"]
        self.eps = exploration_parameters["KernelEpsilon"]
        self.c = exploration_parameters["KernelConstant"]
        self.similarity_max = exploration_parameters["MaximumSimilarity"]
        self.episodic_memory = deque(maxlen=exploration_parameters["EpisodicMemoryCapacity"])
        self.mean_distances = deque(maxlen=self.episodic_memory.maxlen)

        self.feature_extractor, self.embedding_classifier = self.build_network()
        self.episodic_novelty_module_built = True
        # endregion

        # region Lifelong novelty module
        self.normalize_observations = exploration_parameters["ObservationNormalization"]
        self.observation_deque = deque(maxlen=1000)
        self.observation_mean = 0
        self.observation_std = 1
        self.alpha_max = 5
        self.rnd_reward_deque = deque(maxlen=1000)
        self.rnd_reward_mean = 0
        self.rnd_reward_std = 1

        self.prediction_model, self.target_model = self.build_network()
        # endregion

    def build_network(self):
        with tf.device(self.device):
            # region Episodic Novelty Module
            if not self.episodic_novelty_module_built:
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
            # endregion
            # region Lifelong Novelty Module
            else:
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

                return prediction_model, target_model
                # endregion

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
            # region RND
            with tf.GradientTape() as tape:
                target_features = self.target_model(next_state_batch)
                prediction_features = self.prediction_model(next_state_batch)
                self.lifelong_loss = self.mse(target_features, prediction_features)

            # Calculate Gradients and apply the weight updates to the prediction model.
            grad = tape.gradient(self.lifelong_loss, self.prediction_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grad, self.prediction_model.trainable_weights))
            # endregion

            # region ENM
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
                    self.episodic_loss = self.cce(true_actions_one_hot, action_prediction)

                elif self.action_space == "CONTINUOUS":
                    # Compute Loss via Mean Squared Error
                    self.episodic_loss = self.mse(action_batch, action_prediction)

            # Calculate Gradients
            grad = tape.gradient(self.episodic_loss, [self.embedding_classifier.trainable_weights,
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
        config_dict = NeverGiveUp.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        if len(decision_steps.obs[0]):
            current_state = decision_steps.obs
        else:
            self.rnd_reward_deque.append(0)
            self.mean_distances.append(0)
            return 0
        '''
        # Alternative
        else:
            current_state = terminal_steps.obs
        '''

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
        target_features = self.target_model(current_state)
        prediction_features = self.prediction_model(current_state)

        # The rnd reward is the L2 error between target and prediction features summed over all features.
        rnd_reward = tf.math.sqrt(tf.math.reduce_sum(
            tf.math.square(target_features - prediction_features), axis=-1)).numpy()
        rnd_reward = rnd_reward[0]

        # Calculate the running standard deviation and mean of the rnd rewards to normalize it.
        self.rnd_reward_deque.append(rnd_reward)
        self.rnd_reward_std = np.std(self.rnd_reward_deque)
        self.rnd_reward_mean = np.mean(self.rnd_reward_deque)

        # Normalize reward value
        if self.rnd_reward_mean and self.rnd_reward_std:
            rnd_reward = 1 + (rnd_reward - self.rnd_reward_mean) / self.rnd_reward_std
        else:
            rnd_reward = 1
        # endregion

        # region Episodic Novelty Module
        # Extract relevant features from current state
        state_embedding = self.feature_extractor(current_state)

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
            enm_reward = 0
        else:
            # 1/similarity to encourage visiting states with lower similarity
            enm_reward = self.beta * (1 / similarity)
        # endregion

        # Calculate final and combined intrinsic reward
        self.intrinsic_reward = enm_reward * min(max(rnd_reward, 1), self.alpha_max)

        return self.intrinsic_reward


    def get_logs(self):
        return {"Exploration/EpisodicLoss": self.episodic_loss,
                "Exploration/LifeLongLoss": self.lifelong_loss,
                "Exploration/Reward_Act{:02d}_{:.4f}".format(self.index, self.beta): self.intrinsic_reward}

    def reset(self):
        """Empty episodic memory and clear euclidean distance metrics."""
        '''
        self.mean_distances.clear()
        self.episodic_memory.clear()
        '''
        return

    def prevent_checkpoint(self):
        return False
