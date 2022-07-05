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
        self.observation_shapes_modified = observation_shapes

        self.index = idx
        self.device = '/cpu:0'

        self.reset_counter = 0

        # Epsilon-Greedy Parameters
        self.epsilon_decay = exploration_parameters["EpsilonDecay"]
        self.epsilon_min = exploration_parameters["EpsilonMin"]
        self.step_down = exploration_parameters["StepDown"]
        self.epsilon = self.get_epsilon_greedy_parameters(self.index, training_parameters["ActorNum"])
        self.training_step = 0

        # TODO: Use separate function
        # Modify observation shapes to use modified shape for sampling
        modified_observation_shapes = []
        for obs_shape in self.observation_shapes:
            modified_observation_shapes.append(obs_shape)
        modified_observation_shapes.append((self.action_shape,))
        modified_observation_shapes.append((1,))
        modified_observation_shapes.append((1,))
        modified_observation_shapes.append((1,))
        #modified_observation_shapes.append((1,))
        self.observation_shapes_modified = modified_observation_shapes

        # Parameters required during network build-up
        self.episodic_novelty_module_built = False
        self.recurrent = training_parameters["Recurrent"]
        self.sequence_length = training_parameters["SequenceLength"]
        self.feature_space_size = exploration_parameters["FeatureSpaceSize"]

        # Categorical Cross-Entropy for discrete action spaces
        # Mean Squared Error for continuous action spaces
        if self.action_space == "DISCRETE":
            self.cce = CategoricalCrossentropy()
        elif self.action_space == "CONTINUOUS":
            self.mse = MeanSquaredError()

        self.optimizer = Adam(exploration_parameters["LearningRate"])
        self.lifelong_loss = 0
        self.episodic_loss = 0
        self.intrinsic_reward = 0

        # region Episodic novelty module parameters
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

        # region Lifelong novelty module parameters
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
                x = Dense(64, 'relu')(x)
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
            # endregion

            # region Lifelong Novelty Module
            else:
                # region Prediction Model
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
                x = Dense(self.feature_space_size, activation=None)(x)
                prediction_model = Model(feature_input, x, name="RND_PredictionModel")
                # endregion

                # region Target Model
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
                x = Dense(self.feature_space_size, activation=None)(x)
                target_model = Model(feature_input, x, name="RND_TargetModel")
                # endregion

                # region Model plots
                try:
                    plot_model(prediction_model, "plots/RND_PredictionModel.png", show_shapes=True)
                    plot_model(target_model, "plots/RND_TargetModel.png", show_shapes=True)
                except ImportError:
                    print("Could not create model plots for RND.")

                # Summaries
                prediction_model.summary()
                target_model.summary()
                # endregion

                return prediction_model, target_model
            # endregion

    def learning_step(self, replay_batch):
        # region --- Batch Reshaping ---
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes_modified,
                                                                         self.action_shape, self.sequence_length)

            # Only use last 5 time steps of sequences for training
            '''state_batch = [state_input[:, -5:] for state_input in state_batch]
            next_state_batch = [next_state_input[:, -5:] for next_state_input in next_state_batch]
            action_batch = [action_sequence[-5:] for action_sequence in action_batch]'''

        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes_modified,
                                                               self.action_shape)

        if np.any(np.isnan(action_batch)):
            return replay_batch

        # Clear extra state parts added during acting as they must not be used by the exploration algorithms
        state_batch = state_batch[:-4]
        next_state_batch = next_state_batch[:-4]
        # endregion

        # region Epsilon-Greedy learning step
        self.training_step += 1
        if self.epsilon >= self.epsilon_min and not self.step_down:
            self.epsilon *= self.epsilon_decay

        if self.training_step >= self.step_down and self.step_down:
            self.epsilon = self.epsilon_min
        # endregion

        # region Episodic and Lifelong Novelty Modules learning step
        with tf.device(self.device):
            # region Lifelong Novelty Module
            with tf.GradientTape() as tape:
                target_features = self.target_model(next_state_batch)
                prediction_features = self.prediction_model(next_state_batch)
                self.lifelong_loss = self.mse(target_features, prediction_features)

            # Calculate Gradients and apply the weight updates to the prediction model.
            grad = tape.gradient(self.lifelong_loss, self.prediction_model.trainable_weights)
            self.optimizer.apply_gradients(zip(grad, self.prediction_model.trainable_weights))
            # endregion

            # region Episodic Novelty Module
            with tf.GradientTape() as tape:
                # Calculate features of current and next state
                state_features = self.feature_extractor(state_batch)
                next_state_features = self.feature_extractor(next_state_batch)

                action_prediction = self.embedding_classifier([state_features, next_state_features])

                # Calculate inverse loss
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
        # endregion
        return

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = NeverGiveUp.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        if not len(decision_steps.obs[0]):
            current_state = terminal_steps.obs
            '''self.rnd_reward_deque.append(0)
            self.mean_distances.append(0)
            return 0'''
        else:
            current_state = decision_steps.obs

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
        if self.recurrent:
            # Add additional time dimension if recurrent networks are used
            current_state = [tf.expand_dims(state, axis=1) for state in current_state]
            # '[0]' -> to get rid of time dimension directly after network inference
            target_features = self.target_model(current_state)[0]
            prediction_features = self.prediction_model(current_state)[0]
        else:
            target_features = self.target_model(current_state)
            prediction_features = self.prediction_model(current_state)

        # The rnd reward is the L2 error between target and prediction features summed over all features.
        rnd_reward = tf.math.sqrt(tf.math.reduce_sum(
            tf.math.square(target_features - prediction_features), axis=-1)).numpy()
        # To prevent scalar related logging error
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
        if self.recurrent:
            # '[0]' -> to get rid of time dimension directly after network inference
            state_embedding = self.feature_extractor(current_state)[0]
        else:
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
            enm_reward = (1 / similarity)
        # endregion

        # Calculate final and combined intrinsic reward
        self.intrinsic_reward = enm_reward * min(max(rnd_reward, 1), self.alpha_max)

        return self.intrinsic_reward

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
        return {"Exploration/EpisodicLoss": self.episodic_loss,
                "Exploration/LifeLongLoss": self.lifelong_loss}

    def reset(self):
        """Empty episodic memory and clear euclidean distance metrics."""
        self.mean_distances.clear()
        self.episodic_memory.clear()
        return

    def prevent_checkpoint(self):
        return False
