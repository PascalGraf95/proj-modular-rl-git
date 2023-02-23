import numpy as np
from tensorflow import keras
from ..misc.replay_buffer import FIFOBuffer
from .exploration_algorithm_blueprint import ExplorationAlgorithm
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
from ..misc.network_constructor import construct_network
import tensorflow as tf
from ..misc.utility import modify_observation_shapes
from ..training_algorithms.agent_blueprint import Learner
import itertools
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
import time
from collections import deque
from tensorflow.keras.utils import plot_model


class NeverGiveUpReach(ExplorationAlgorithm):
    """
    Alternative implementation of Agent57's reward generator NeverGiveUp (NGU). The algorithm "Episodic Curiosity
    through Reachability" (ECR) is used here instead of the "Episodic Novelty Module" (ENM).
    The computation of intrinsic episodic rewards is done for each actor and after every environment step (see act()).

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

        # Categorical Cross-Entropy for ECR, MSE is used within the life-long module.
        self.cce = CategoricalCrossentropy()
        self.mse = MeanSquaredError()
        # endregion

        # region - Misc -
        self.index = idx
        self.device = '/cpu:0'
        self.training_step = 0
        self.episodic_loss = 0
        self.lifelong_loss = 0
        self.intrinsic_reward = 0
        self.episodic_curiosity_module_built = False
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
        self.optimizer_episodic = Adam(learning_rate=0.00005)
        # self.optimizer_episodic = Adam(exploration_parameters["LearningRate"])
        self.optimizer_lifelong = Adam(parameters["LearningRate"])
        # endregion

        # region - Episodic Curiosity Module specific Parameters -
        # k gives the threshold for the determination if states are reachable/not reachable one from each other. As this
        # is measured in environmental steps, k represents a number of steps.
        self.k = parameters["kECR"]
        # Fixed value, should be set as follows -> For envs with fixed-length episodes: 0.5 else 1.0
        self.beta = parameters["BetaECR"]
        # For an overall scaling of the intrinsic reward, paper suggests a value of '1.0'. Higher values lead to higher
        # intrinsic rewards.
        self.alpha = 1.0
        # Calculate left and right limits of the value range of the overall output and get median, which represents the
        # novelty threshold.
        min_network_output = 0  # as output node of comparator is sigmoid
        max_network_output = 1  # as output node of comparator is sigmoid
        left_limit, right_limit = self.alpha * (self.beta - min_network_output), \
                                  self.alpha * (self.beta - max_network_output)
        # A value of '0' for the novelty threshold is suggested by the authors of the paper. However, combined with
        # self.beta = 1.0 (-> Envs with variable episode lengths) this leads to every single state encountered being
        # added to the episodic memory. Therefore, the median of the value range is used for the novelty threshold.
        # Further explanations can be found in the respective thesis documentation (Agent57: State-Of-The-Art Deep
        # Reinforcement Learning).
        # original paper: self.novelty_threshold = 0
        self.novelty_threshold = np.median([left_limit, right_limit])
        # Episodic memory cleared at every episode end.
        self.episodic_memory = deque(maxlen=parameters["EpisodicMemoryCapacity"])
        # Constants necessary for the training data calculations during runtime
        self.sequence_indices = np.arange(self.sequence_length)
        self.sequence_middle = self.sequence_length // 2
        # endregion

        # region - Lifelong Novelty Module specific Parameters -
        self.normalize_observations = parameters["ObservationNormalization"]
        self.observation_deque = deque(maxlen=1000)
        self.observation_mean = 0
        self.observation_std = 1
        self.alpha_max = 5
        self.lifelong_reward_deque = deque(maxlen=1000)
        self.lifelong_reward_mean = 0
        self.lifelong_reward_std = 1
        # endregion

        # region - Modify observation shapes and construct neural networks-
        # Takes into account that prior actions, rewards and policies might be used for state augmentation.
        # However, their values should not be considered for ENM. The modified observation shape is only used for
        # sample batch preprocessing.
        self.observation_shapes_modified = modify_observation_shapes(self.observation_shapes, self.action_shape,
                                                                     self.action_space,
                                                                     trainer_configuration["ActionFeedback"],
                                                                     trainer_configuration["RewardFeedback"],
                                                                     trainer_configuration["PolicyFeedback"])
        self.num_additional_obs_values = len(self.observation_shapes_modified) - len(self.observation_shapes)
        # ECR utilizes two network architectures, a feature extractor which converts the provided state into a fixed
        # length feature vector and a comparator which takes two state embeddings as input to predict if they are
        # reachable or non-reachable one from each other within k-steps. LNM utilizes also two network architectures,
        # a prediction model which tries to predict the outputs of a target model. Overall the LNM output is used as a
        # modulator for up- or downscaling the ECR output.
        # Episodic Novelty Module Networks
        self.feature_extractor, self.comparator_network = self.build_network()
        self.episodic_curiosity_module_built = True
        # Lifelong Novelty Module Networks
        self.prediction_model, self.target_model = self.build_network()
        # endregion

    def build_network(self):
        with tf.device(self.device):
            # region - Episodic Curiosity Module -
            if not self.episodic_curiosity_module_built:
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
                feature_extractor = Model(feature_input, x, name="ECR Feature Extractor")
                # endregion

                # region - Comparator -
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
                x = Dense(32, activation="relu")(x)
                x = Dense(2, activation='softmax')(x)
                comparator_network = Model([current_state_features, next_state_features], x, name="ECR Comparator")
                # endregion

                # region - Model Plots and Summaries -
                try:
                    plot_model(feature_extractor, "plots/ECR_Feature_Extractor.png", show_shapes=True)
                    plot_model(comparator_network, "plots/ECR_Comparator.png", show_shapes=True)
                except ImportError:
                    print("Could not create model plots for ECR.")

                # Summaries
                feature_extractor.summary()
                comparator_network.summary()
                # endregion

                return feature_extractor, comparator_network
            # endregion

            # region - Lifelong Novelty Module -
            else:
                # region - Prediction Model -
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
                prediction_model = Model(feature_input, x, name="RND Prediction Model")
                # endregion

                # region - Target Model -
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
                target_model = Model(feature_input, x, name="RND Target Model")
                # endregion

                # region - Model plots -
                try:
                    plot_model(prediction_model, "plots/RND_Prediction_Model.png", show_shapes=True)
                    plot_model(target_model, "plots/RND_Target_Model.png", show_shapes=True)
                except ImportError:
                    print("Could not create model plots for RND.")

                # Summaries
                prediction_model.summary()
                target_model.summary()
                # endregion

                return prediction_model, target_model
            # endregion

    def learning_step(self, replay_batch):
        # region - Batch Reshaping -
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes_modified,
                                                                         self.action_shape, self.sequence_length)
            # Only use last 5 time steps of sequences for training
            '''state_batch = [state_input[:, -5:] for state_input in state_batch]
            next_state_batch = [next_state_input[:, -5:] for next_state_input in next_state_batch]
            action_batch = action_batch[:, -5:]'''
        else:
            raise RuntimeError("Exploration algorithm 'NGU' does currently not work with non-recurrent agents.")

        if np.any(np.isnan(action_batch)):
            return replay_batch

        # Clear augmented state parts such as rewards, actions and policy indices as they must not be used by
        # the exploration algorithms, i.e. closeness of states should not depend on the intrinsic reward or the
        # exploration strategy followed at that time. Furthermore, providing the taken action would make the problem
        # trivial.
        if self.num_additional_obs_values:
            state_batch = state_batch[:-self.num_additional_obs_values]
            next_state_batch = next_state_batch[:-self.num_additional_obs_values]

        # region - Epsilon-Greedy learning step -
        # Decrease epsilon gradually until epsilon min is reached.
        self.training_step += 1
        if self.epsilon >= self.epsilon_min and not self.step_down:
            self.epsilon *= self.epsilon_decay
        # After playing with high epsilon for a certain number of episodes, abruptly decrease epsilon to its min value.
        if self.training_step >= self.step_down and self.step_down:
            self.epsilon = self.epsilon_min
        # endregion

        # region - Training -
        # Classifier and feature extractor are trained end-to-end on pairs of states and next states.
        # The feature extractor embeds both states, the classifier predicts which action lead to the respective
        # transition. This way it is ensured that the feature extractor only considers features and patterns that
        # play a role for determining the agent's action.
        with tf.device(self.device):
            # region - LNM -
            # The Prediction Model is trained on the mse between the predicted values of the Target and Prediction
            # Model. The Target Model is not updated.
            with tf.GradientTape() as tape:
                # Calculate features
                target_features = self.target_model(next_state_batch)
                prediction_features = self.prediction_model(next_state_batch)

                # Compute Loss via Mean Squared Error
                self.lifelong_loss = self.mse(target_features, prediction_features)

            # Calculate Gradients and apply the weight updates to the prediction model.
            grad = tape.gradient(self.lifelong_loss, self.prediction_model.trainable_weights)
            self.optimizer_lifelong.apply_gradients(zip(grad, self.prediction_model.trainable_weights))
            # endregion

            # Comparator is trained on pairs of state embeddings with respective labels that give the info if the
            # compared states are reachable or non-reachable with regard to the threshold k. However, after determining
            # the embedded states, the training data (random but unique pairs of embedded states with respective
            # reachability labels) needs to be calculated in the first place. The comparator then predicts the
            # reachability between the state pairs and is trained afterwards based on the categorical cross-entropy loss
            # given through the comparison between the predicted and the priorly manually determined reachability
            # labels. For performance reasons only the comparator is trained, which, according to the original paper,
            # does not lead to critical performance degradations.
            with tf.GradientTape() as tape:
                # Calculate features
                state_features = self.feature_extractor(state_batch)

                # Create training data (unique feature-pairs with reachability information as labels)
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

                # Compute loss via Categorical Cross-Entropy
                self.episodic_loss = self.cce(y_true_batch, y_pred)

            # Calculate Gradients
            grad = tape.gradient(self.episodic_loss, self.comparator_network.trainable_weights)
            # Apply Gradients only to the comparator model.
            self.optimizer_episodic.apply_gradients(zip(grad, self.comparator_network.trainable_weights))
        # endregion
        return

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = NeverGiveUpReach.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        """
        Calculate intrinsically-based episodic reward through reachability comparison of current state embedding with
        the content of the episodic memory and afterwards modulating through lifelong module output.
        """
        if not len(decision_steps.obs[0]):
            current_state = terminal_steps.obs
        else:
            current_state = decision_steps.obs

        # region - Lifelong Novelty Module -
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
        lifelong_intrinsic_reward = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(target_features - prediction_features),
                                                                    axis=-1)).numpy()
        # To prevent scalar related logging error
        lifelong_intrinsic_reward = lifelong_intrinsic_reward[0]

        # Calculate the running standard deviation and mean of the rnd rewards to normalize it.
        self.lifelong_reward_deque.append(lifelong_intrinsic_reward)
        self.lifelong_reward_std = np.std(self.lifelong_reward_deque)
        self.lifelong_reward_mean = np.mean(self.lifelong_reward_deque)

        # Normalize reward value
        if self.lifelong_reward_mean and self.lifelong_reward_std:
            lifelong_intrinsic_reward = 1 + (lifelong_intrinsic_reward - self.lifelong_reward_mean) / self.lifelong_reward_std
        else:
            lifelong_intrinsic_reward = 1
        # endregion

        # region - Episodic Novelty Module -
        # region - Feature embedding -
        if self.recurrent:
            """# Add additional time dimension if recurrent networks are used
            current_state = [tf.expand_dims(state, axis=1) for state in current_state]"""
            # Calculate state embedding and get rid of additional time dimension through index 0
            state_embedding = self.feature_extractor(current_state)[0]
        else:
            print("Exploration algorithm 'ECR' does not work with non-recurrent agents. Acting step NOT executed.")
            return 0
        # endregion

        # region - Comparison preparations -
        # First state embedding must be added to episodic memory before executing further calculations
        if not self.episodic_memory.__len__():
            self.episodic_memory.append(state_embedding)
            return 0

        # Create array with length of the current episodic memory containing same copies of the current state embedding
        state_embedding_array = np.empty([self.episodic_memory.__len__(), state_embedding.shape[0],
                                          state_embedding.shape[1]])
        # Place the current state embedding into every array row to allow for a '1-to-1' comparison with respect to all
        # content within episodic memory.
        state_embedding_array[:] = state_embedding
        # endregion

        # region - Reachability buffer determination and aggregation -
        # Reachability buffer contains values from 0...1 indicating if compared states are reachable one from each other
        # (==1).
        reachability_buffer = self.comparator_network([state_embedding_array, np.array(self.episodic_memory)])[:, :, 1]

        # Aggregate the content of the reachability buffer to calculate a similarity-score of the current embedding.
        # Gives a single value that indicates the similarity of the current embedding to the already seen states within
        # the episode.
        similarity_score = np.percentile(reachability_buffer, 90)
        # endregion

        # region - Calculate the episodic intrinsic reward and check for state novelty -
        # 𝑟 = 𝛼 ∗ (𝛽 − 𝑠𝑖𝑚𝑖𝑙𝑎𝑟𝑖𝑡𝑦_𝑠𝑐𝑜𝑟𝑒)
        episodic_intrinsic_reward = self.alpha * (self.beta - similarity_score)

        # States must only be added to episodic memory if intrinsic reward is large enough/the state embedding is
        # relevant enough for the exploration process.
        if episodic_intrinsic_reward > self.novelty_threshold:
            self.episodic_memory.append(state_embedding)
        # endregion
        # endregion

        # region - Final intrinsic reward calculation -
        # Modulate episodic curiosity module output by lifelong novelty module output
        self.intrinsic_reward = episodic_intrinsic_reward * min(max(lifelong_intrinsic_reward, 1), self.alpha_max)
        # endregion

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
        Create training data through forming random embedded state pairs and calculating whether the elements of
        those pairs are reachable one from each other within k-steps. The comparator model predicts two classes with
        the first one being "non-reachable" and the second one being "reachable". Therefore, the returned labels within
        this function consist of a list with two elements, where the first one indicates the probability that the states
        ARE NOT reachable and the second one the probability that the states ARE reachable. This gives the label
        formulation:
            -> non-reachable:  [1,0]
            -> reachable:      [0,1]

        ***NOTE***
        Generation of observation pairs is done randomly and differs from the original paper where a sliding window
        based approach is used that does not fit here, as it generates random amounts of pairs each iteration.

        Parameters
        ----------
        sequence:
            Contains the embedded observation values of a sequence from the state_batch.

        Returns
        -------
        x1:
            First elements of the embedded observation index pairs.
        x2:
            Second elements of the embedded observation index pairs.
        labels: list
            Reachability between x1 and x2 elements. ([1, 0] == not reachable, [0, 1] == reachable)
        """
        # Shuffle sequence indices randomly
        np.random.shuffle(self.sequence_indices)

        # Divide Index-Array into two equally sized parts
        sequence_indices_left, sequence_indices_right = self.sequence_indices[:self.sequence_middle], \
                                                        self.sequence_indices[
                                                        self.sequence_middle:2 * self.sequence_middle]
        idx_differences = np.abs(sequence_indices_left - sequence_indices_right)
        x1 = sequence.numpy()[sequence_indices_left]
        x2 = sequence.numpy()[sequence_indices_right]

        # States are reachable one from each other if step-difference between them is smaller than k
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
