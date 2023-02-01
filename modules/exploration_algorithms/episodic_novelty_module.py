import numpy as np
from tensorflow import keras
from ..misc.replay_buffer import FIFOBuffer
from .exploration_algorithm_blueprint import ExplorationAlgorithm
from keras.losses import MeanSquaredError, CategoricalCrossentropy
from keras.optimizers import Adam, SGD
from ..misc.network_constructor import construct_network
import tensorflow as tf
from ..misc.utility import modify_observation_shapes
from ..training_algorithms.agent_blueprint import Learner
import itertools
from keras import Input, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
import time
from collections import deque
from keras.utils import plot_model


class EpisodicNoveltyModule(ExplorationAlgorithm):
    """
    Basic implementation of Episodic Novelty Module (ENM)
    The computation of intrinsic episodic rewards is done for each actor and after every environment step (see act()).

    https://openreview.net/pdf?id=Sye57xStvB
    """
    Name = "EpisodicNoveltyModule"
    ActionAltering = False
    IntrinsicReward = True

    def __init__(self, action_shape, observation_shape,
                 action_space,
                 parameters,
                 trainer_configuration, idx):
        # region - Action and Observation Spaces -
        # The type of action space and shape determines the loss function for training feature extractor and embedding
        # classifier. Also, differences in shapes between DISCRETE and CONTINUOUS action spaces have to be considered.
        self.action_space = action_space
        self.action_shape = action_shape
        # The observation shapes have to be known for network construction. Furthermore, they might be modified
        # by the augmentation of other metrics via feedback.
        self.observation_shapes = observation_shape
        self.observation_shapes_modified = self.observation_shapes

        # Categorical Cross-Entropy for discrete action spaces, Mean Squared Error for continuous action spaces.
        if self.action_space == "DISCRETE":
            self.cce = CategoricalCrossentropy()
        elif self.action_space == "CONTINUOUS":
            self.mse = MeanSquaredError()
        # endregion

        # region - Misc -
        self.index = idx
        self.device = '/cpu:0'
        self.training_step = 0
        self.loss = 0
        self.episodic_intrinsic_reward = 0
        # endregion

        # region - Epsilon-Greedy and Network Parameters -
        # Parameters as for vanilla Epsilon Greedy, decay determining the exponential decreasing of exploration,
        # min value determining the minimum randomness and step down enabling acting randomly for a given number of
        # episodes.
        self.epsilon_decay = parameters["EpsilonDecay"]
        self.epsilon_min = parameters["EpsilonMin"]
        self.step_down = parameters["StepDown"]
        self.epsilon = self.get_epsilon_greedy_parameters(self.index, trainer_configuration["ActorNum"])

        # Recurrent parameters determine the dimensionality of replay batches.
        self.recurrent = trainer_configuration["Recurrent"]
        self.sequence_length = trainer_configuration["SequenceLength"]
        # The chosen Feature Space Size corresponds to the number of output neurons in the feature extractor and thus
        # determines how much the original state is compressed.
        self.feature_space_size = parameters["FeatureSpaceSize"]
        self.optimizer = Adam(parameters["LearningRate"])

        # endregion

        # region - Episodic Novelty Module specific Parameters -
        # K determines how many closest states are considered during calculation of the similarity of a new state to
        # older states in the same episode.
        self.k = parameters["kNearest"]
        # The cluster distance defines the maximum distance in embedded space in which two points are considered to
        # belong to the same cluster.
        self.cluster_distance = parameters["ClusterDistance"]
        # The kernel of the similarity function eps / (x + eps).
        self.eps = parameters["KernelEpsilon"]
        # Small positive float to prevent division by zero.
        self.c = 0.001
        # Maximum similarity threshold over which no intrinsic episode reward is assigned.
        self.similarity_max = parameters["MaximumSimilarity"]
        # Episodic memory and mean distance deque cleared at every episode end.
        self.episodic_memory = deque(maxlen=parameters["EpisodicMemoryCapacity"])
        self.mean_distances = deque(maxlen=self.episodic_memory.maxlen)
        # endregion

        # region - Modify observation shapes and construct Neural Networks -
        # Takes into account that prior actions, rewards and policies might be used for state augmentation.
        # However, their values should not be considered for ENM. The modified observation shape is only used for
        # sample batch preprocessing.
        self.observation_shapes_modified = modify_observation_shapes(self.observation_shapes, self.action_shape,
                                                                     self.action_space,
                                                                     trainer_configuration["ActionFeedback"],
                                                                     trainer_configuration["RewardFeedback"],
                                                                     trainer_configuration["PolicyFeedback"])
        self.num_additional_obs_values = len(self.observation_shapes_modified) - len(self.observation_shapes)
        # ENM utilizes two network architectures, a feature extractor which converts the provided state into a fixed
        # length feature vector and an embedding classifier which takes two subsequent state embeddings as input to
        # predict which action that lead to the transition. Its output is utilized for training both networks similar
        # to ICM.
        self.feature_extractor, self.embedding_classifier = self.build_network()
        # endregion

    def build_network(self):
        with tf.device(self.device):
            # region - Feature Extractor -
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

            # region - Embedding Classifier -
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

            # region - Model Plots and Summaries -
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

    def learning_step(self, replay_batch):
        # region - Batch Reshaping -
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes_modified,
                                                                         self.action_shape, self.sequence_length)

        else:
            raise RuntimeError("Exploration algorithm 'ENM' does currently not work with non-recurrent agents.")

        if np.any(np.isnan(action_batch)):
            return replay_batch

        # Clear augmented state parts such as rewards, actions and policy indices as they must not be used by
        # the exploration algorithms, i.e. closeness of states should not depend on the intrinsic reward or the
        # exploration strategy followed at that time. Furthermore, providing the taken action would make the problem
        # trivial.
        if self.num_additional_obs_values:
            state_batch = state_batch[:-self.num_additional_obs_values]
            next_state_batch = next_state_batch[:-self.num_additional_obs_values]

        # The action batch, if discrete, contains the index of the respective actions multiple times for every step,
        # which is not necessary for further operations, therefore get only the first element for every sequence.
        if self.action_space == "DISCRETE":
            action_batch = action_batch[:, :, 0]
        # endregion

        # region - Epsilon-Greedy learning step -
        # Decrease epsilon gradually until epsilon min is reached.
        self.training_step += 1
        if self.epsilon >= self.epsilon_min and not self.step_down:
            self.epsilon *= self.epsilon_decay
        # After playing with high epsilon for a certain number of episodes, abruptly decrease epsilon to its min value.
        if self.training_step >= self.step_down and self.step_down:
            self.epsilon = self.epsilon_min
        # endregion

        # region - Embedding Classifier and  Feature Extractor training -
        # Classifier and feature extractor are trained end-to-end on pairs of states and next states.
        # The feature extractor embeds both states, the classifier predicts which action lead to the respective
        # transition. This way it is ensured that the feature extractor only considers features and patterns that
        # play a role for determining the agent's action.
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                # Calculate features of the current and the subsequent state
                state_features = self.feature_extractor(state_batch)
                next_state_features = self.feature_extractor(next_state_batch)

                # Predict which actions led to the transitions based on extracted features
                action_prediction = self.embedding_classifier([state_features, next_state_features])

                # Calculate Inverse Loss
                if self.action_space == "DISCRETE":
                    # Encode true action as one hot vector encoding
                    true_actions_one_hot = tf.one_hot(action_batch, self.action_shape[0])
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
        # endregion
        return

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = EpisodicNoveltyModule.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        """
        Calculate intrinsically-based episodic reward through similarity comparison of current state embedding with
        the content of the episodic memory.
        """
        if not len(decision_steps.obs[0]):
            current_state = terminal_steps.obs
        else:
            current_state = decision_steps.obs

        # region - Feature embedding and distance calculation -
        if self.recurrent:
            # Add additional time dimension if recurrent networks are used
            current_state = [tf.expand_dims(state, axis=1) for state in current_state]
            # Calculate state embedding and get rid of additional time dimension through index 0
            state_embedding = self.feature_extractor(current_state)[0]
        else:
            state_embedding = self.feature_extractor(current_state)

        # Calculate the euclidean distances between current state embedding and the ones within episodic memory.
        # The episodic memory is cleared at the end of each episode as the name suggests.
        embedding_distances = [np.linalg.norm(mem_state_embedding - state_embedding)
                               for mem_state_embedding in self.episodic_memory]
        # Add the state embedding to episodic memory
        self.episodic_memory.append(state_embedding)
        # endregion

        # region - Top K distances determination and Normalization -
        # Get a list of the top k smallest distances between the current state embedding and the episodic memory.
        top_k_emb_distances = np.sort(embedding_distances)[:self.k]  # ascending order
        # Calculate the mean distance between the top k embeddings and the current state embedding.
        # If the array is empty, this is the first step in the episode. In this case return.
        if np.any(top_k_emb_distances):
            self.mean_distances.append(np.mean(top_k_emb_distances))
        else:
            # Mean distance will be one for first iteration, as episodic memory is empty
            self.mean_distances.append(1)
            return 0

        # Normalize the top k distances with a moving average of mean distances from previous episode steps
        top_k_emb_distances_normalized = top_k_emb_distances / np.mean(self.mean_distances)
        # endregion

        # region - Clustering and Similarity Calculation -
        # Only keep top k distances which are greater than a certain distance, i.e. distances to prior states which thus
        # do not belong to the same cluster. Rest becomes 0.
        top_k_emb_distances_clipped = np.where(top_k_emb_distances_normalized - self.cluster_distance >= 0,
                                               top_k_emb_distances_normalized - self.cluster_distance, 0)

        # Calculate the Similarity
        # k is the kernel epsilon (a small float) divided by the top_k_distances elementwise for each distance + the
        # kernel epsilon.
        # For state embeddings within the same cluster (i.e. small distances) where the clipped value is 0, k is one.
        # For greater distances k approaches 0.
        k = self.eps / (top_k_emb_distances_clipped + self.eps)
        # To get the similarity all elements of k are summed and the square root is taken. Also, a small positive value
        # is added to prevent division errors later.
        # The more unique the new state embedding is, the smaller k and the smaller the similarity which will lead
        # to a higher intrinsic reward.
        similarity = np.sqrt(np.sum(k)) + self.c

        # If the similarity exceeds a certain threshold there will be no intrinsic episodic reward.
        if np.isnan(similarity) or (similarity > self.similarity_max):
            self.episodic_intrinsic_reward = 0
        else:
            # Otherwise, the intrinsic episodic reward is calculated 1/similarity to encourage visiting
            # states with lower similarity (via extended observation space).
            self.episodic_intrinsic_reward = (1 / similarity)
        # endregion
        return self.episodic_intrinsic_reward

    def epsilon_greedy(self, decision_steps):
        if len(decision_steps.agent_id):
            if np.random.rand() < self.epsilon:
                if self.action_space == "DISCRETE":
                    return np.random.randint(0, self.action_shape, (len(decision_steps.agent_id), 1))
                else:
                    return np.random.uniform(-1.0, 1.0, (len(decision_steps.agent_id), self.action_shape))
        return None

    def get_epsilon_greedy_parameters(self, actor_idx, num_actors):
        """
        Calculate this actors' epsilon for epsilon-greedy acting behaviour.

        Parameters
        ----------
        actor_idx: int
            The index of this actor.
        num_actors: int
            Total number of actors used.

        Returns
        -------
        epsilon: float
            Epsilon-Greedy's initial epsilon value.
        """
        epsilon = 0.4**(1 + 8 * (actor_idx / (num_actors - 1)))
        if epsilon < self.epsilon_min:
            epsilon = self.epsilon_min
        return epsilon

    def get_logs(self):
        if self.index == 0:
            return {"Exploration/EpisodicLoss": self.loss,
                    "Exploration/IntrinsicReward": self.episodic_intrinsic_reward}
        else:
            return {"Exploration/EpisodicLoss": self.loss}

    def reset(self):
        """Empty episodic memory and clear euclidean distance metrics."""
        self.mean_distances.clear()
        self.episodic_memory.clear()
        return

    def prevent_checkpoint(self):
        return False
