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


class NeverGiveUpReach(ExplorationAlgorithm):
    """
    Basic implementation of NeverGiveUp's intrinsic reward generator but with the algorithm 'Episodic Curiosity through
    reachability' instead of the 'Episodic Novelty Module'. (-> Incorporates ECR and RND)

    NGU: https://openreview.net/pdf?id=Sye57xStvB
    ECR: https://arxiv.org/abs/1810.02274
    """
    Name = "NeverGiveUpReach"
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

        # Modify observation shapes for sampling later on
        self.observation_shapes_modified = modify_observation_shapes(self.observation_shapes, self.action_shape,
                                                                     self.action_space,
                                                                     training_parameters["ActionFeedback"],
                                                                     training_parameters["RewardFeedback"],
                                                                     training_parameters["PolicyFeedback"])
        self.num_additional_obs_values = len(self.observation_shapes_modified) - len(self.observation_shapes)

        # Parameters required during network build-up
        self.episodic_curiosity_built = False
        self.recurrent = training_parameters["Recurrent"]
        self.sequence_length = training_parameters["SequenceLength"]
        self.batch_size = training_parameters["BatchSize"]
        self.feature_space_size = exploration_parameters["FeatureSpaceSize"]

        # Categorical Cross-Entropy
        # Mean Squared Error for Lifelong Novelty Module
        self.cce = CategoricalCrossentropy()
        self.mse = MeanSquaredError()

        self.optimizer = Adam(exploration_parameters["LearningRate"])
        self.lifelong_loss = 0
        self.episodic_loss = 0
        self.intrinsic_reward = 0

        # region ECR parameters
        self.k = exploration_parameters["kECR"]
        self.beta = exploration_parameters["BetaECR"]  # For envs with fixed-length episodes: 0.5 else 1.0
        self.alpha = 10.0

        # Calculate left and right limits of the value range of the output and get median, which represents the novelty
        # threshold
        min_network_output = 0  # as output node of comparator is sigmoid
        max_network_output = 1  # as output node of comparator is sigmoid
        left_limit, right_limit = self.alpha * (self.beta - min_network_output), \
                                  self.alpha * (self.beta - max_network_output)

        # A value of '0' for the novelty threshold is suggested by the authors of the original paper. However, combined
        # with self.beta = 1.0 this leads to every single state encountered being added to the episodic memory. This
        # results in a larger training time overall. Therefore, the median of the value range is used for the novelty
        # threshold.
        # self.novelty_threshold = 0
        self.novelty_threshold = np.median([left_limit, right_limit])

        self.episodic_memory = deque(maxlen=exploration_parameters["EpisodicMemoryCapacity"])
        self.sequence_indices = np.arange(self.sequence_length)
        self.sequence_middle = self.sequence_length // 2

        self.feature_extractor, self.comparator_network = self.build_network()
        self.episodic_curiosity_built = True
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
            # region Episodic Curiosity Module
            if not self.episodic_curiosity_built:
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
                x = Dense(2, activation='softmax')(x)
                comparator_network = Model([current_state_features, next_state_features], x, name="ECR Comparator")
                # endregion

                # region Model compilation and plotting
                comparator_network.compile(loss=self.cce, optimizer=self.optimizer)

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
            action_batch = action_batch[:, -5:]'''

        else:
            return "Exploration algorithm 'NGUr' does currently not work with non-recurrent agents."

        if np.any(np.isnan(action_batch)):
            return replay_batch
        # endregion

        # Epsilon-Greedy learning step
        self.training_step += 1
        if self.epsilon >= self.epsilon_min and not self.step_down:
            self.epsilon *= self.epsilon_decay

        if self.training_step >= self.step_down and self.step_down:
            self.epsilon = self.epsilon_min

        # Clear additional observation parts added as they must not be used by the exploration algorithms
        if self.num_additional_obs_values:
            state_batch = state_batch[:-self.num_additional_obs_values]
            next_state_batch = next_state_batch[:-self.num_additional_obs_values]

        # region --- learning step ---
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

            # region Episodic Curiosity Through Reachability
            with tf.GradientTape() as tape:
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

                # Calculate Categorical Cross-Entropy Loss
                self.episodic_loss = self.cce(y_true_batch, y_pred)

            # Calculate Gradients and apply the weight updates to the comparator model.
            grad = tape.gradient(self.episodic_loss, self.comparator_network.trainable_weights)
            self.optimizer.apply_gradients(zip(grad, self.comparator_network.trainable_weights))

            # endregion
        # endregion
        return

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = NeverGiveUpReach.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        if not len(decision_steps.obs[0]):
            current_state = terminal_steps.obs
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

        # region Episodic Curiosity Through Reachability
        # Extract relevant features from current state
        if self.recurrent:
            # '[0]' -> to get rid of time dimension directly after network inference
            state_embedding = self.feature_extractor(current_state)[0]
        else:
            state_embedding = self.feature_extractor(current_state)

        # First observation must be added to episodic memory before executing further calculations
        if not self.episodic_memory.__len__():
            self.episodic_memory.append(state_embedding)
            return 0

        # Create array with length of the current episodic memory containing same copies of the current state embedding
        state_embedding_array = np.empty([self.episodic_memory.__len__(), state_embedding.shape[0],
                                          state_embedding.shape[1]])
        state_embedding_array[:] = state_embedding

        # Get reachability buffer
        reachability_buffer = self.comparator_network([state_embedding_array, np.array(self.episodic_memory)])[:, :, 1]

        # Aggregate the content of the reachability buffer to calculate similarity-score of current embedding
        similarity_score = np.percentile(reachability_buffer, 90)

        # Calculate ecr-reward
        # 𝑟 = 𝛼 ∗ (𝛽 − 𝑠𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦_𝑠𝑐𝑜𝑟𝑒), where 𝛼 is a simple reward scaling factor that is set outside this module
        ecr_reward = self.alpha * (self.beta - similarity_score)

        # Add state to episodic memory if intrinsic reward is large enough
        if ecr_reward > self.novelty_threshold:
            self.episodic_memory.append(state_embedding)
        # endregion

        # Modulate episodic novelty module output by lifelong novelty module output
        self.intrinsic_reward = ecr_reward * min(max(rnd_reward, 1), self.alpha_max)

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

    def get_training_data(self, sequence):
        """
        Create ECR's training data through forming of random observation pairs and calculating whether the elements of those
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
        # Shuffle sequence indices randomly
        np.random.shuffle(self.sequence_indices)

        # Divide Index-Array into two equally sized parts (Right half gets cutoff if sequence length is odd)
        sequence_indices_left, sequence_indices_right = self.sequence_indices[:self.sequence_middle], \
                                                        self.sequence_indices[
                                                        self.sequence_middle:2 * self.sequence_middle]
        idx_differences = np.abs(sequence_indices_left - sequence_indices_right)
        x1 = sequence.numpy()[sequence_indices_left]
        x2 = sequence.numpy()[sequence_indices_right]

        # States are reachable (== [0, 1]) one from each other if step-difference between them is smaller than k
        diffs = [[0, 1] if diff <= self.k else [1, 0] for diff in idx_differences]

        return x1, x2, diffs

    def get_logs(self):
        if self.index == 0:
            return {"Exploration/EpisodicLoss": self.episodic_loss,
                    "Exploration/LifeLongLoss": self.lifelong_loss,
                    "Exploration/IntrinsicReward": self.intrinsic_reward}
        else:
            return {"Exploration/EpisodicLoss": self.episodic_loss,
                    "Exploration/LifeLongLoss": self.lifelong_loss}

    def reset(self):
        """Empty episodic memory and clear euclidean distance metrics."""
        self.episodic_memory.clear()
        return

    def prevent_checkpoint(self):
        return False
