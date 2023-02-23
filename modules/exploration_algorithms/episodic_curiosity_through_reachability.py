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

    def __init__(self, action_shape, observation_shape,
                 action_space,
                 parameters,
                 trainer_configuration, idx):
        # region - Action and Observation Spaces -
        # The type of action space and shape determines the loss function for training feature extractor and embedding
        # classifier.
        self.action_space = action_space
        self.action_shape = action_shape
        # The observation shapes have to be known for network construction. Furthermore, they might be modified
        # by the augmentation of other metrics via feedback.
        self.observation_shapes = observation_shape
        self.observation_shapes_modified = self.observation_shapes

        self.cce = CategoricalCrossentropy()
        # endregion

        # region - Misc -
        self.index = idx
        self.device = '/cpu:0'
        self.training_step = 0
        self.loss = 0
        self.episodic_intrinsic_reward = 0
        # endregion

        # Recurrent parameters determine the dimensionality of replay batches.
        self.recurrent = trainer_configuration["Recurrent"]
        self.sequence_length = trainer_configuration["SequenceLength"]
        # The chosen Feature Space Size corresponds to the number of output neurons in the feature extractor and thus
        # determines how much the original state is compressed.
        self.feature_space_size = parameters["FeatureSpaceSize"]
        self.optimizer = Adam(parameters["LearningRate"])

        # region - Episodic Curiosity through Reachability specific Parameters -
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

        # ECR utilizes two network architectures, a feature extractor which converts the provided state into a fixed
        # length feature vector and a comparator which takes two state embeddings as input to predict if they are
        # reachable or non-reachable one from each other within k-steps.
        self.feature_extractor, self.comparator_network = self.build_network()
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

    def learning_step(self, replay_batch):
        # region - Batch Reshaping -
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes_modified,
                                                                         self.action_shape, self.sequence_length)

        else:
            raise RuntimeError("Exploration algorithm 'ECR' does currently not work with non-recurrent agents.")

        if np.any(np.isnan(action_batch)):
            return replay_batch

        # Clear augmented state parts such as rewards, actions and policy indices as they must not be used by
        # the exploration algorithms, i.e. closeness of states should not depend on the intrinsic reward or the
        # exploration strategy followed at that time. Furthermore, providing the taken action would make the problem
        # trivial.
        if self.num_additional_obs_values:
            state_batch = state_batch[:-self.num_additional_obs_values]
        # endregion

        # region - Comparator training -
        # Comparator is trained on pairs of state embeddings with respective labels that give the info if the compared
        # states are reachable or non-reachable with regard to the threshold k. However, after determining the embedded
        # states, the training data (random but unique pairs of embedded states with respective reachability labels)
        # needs to be calculated in the first place. The comparator then predicts the reachability between the state
        # pairs and is trained afterwards based on the categorical cross-entropy loss given through the comparison
        # between the predicted and the priorly manually determined reachability labels.
        # For performance reasons only the comparator is trained, which, according to the original paper, does not lead
        # to critical performance degradation.
        with tf.device(self.device):
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
                self.loss = self.cce(y_true_batch, y_pred)

            # Calculate Gradients
            grad = tape.gradient(self.loss, self.comparator_network.trainable_weights)
            # Apply Gradients only to the comparator model.
            self.optimizer.apply_gradients(zip(grad, self.comparator_network.trainable_weights))
        # endregion
        return

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

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    @staticmethod
    def get_config():
        config_dict = EpisodicCuriosity.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        """
        Calculate intrinsically-based episodic reward through reachability comparison of current state embedding with
        the content of the episodic memory.
        """
        if not len(decision_steps.obs[0]):
            current_state = terminal_steps.obs
        else:
            current_state = decision_steps.obs

        # region - Feature embedding -
        if self.recurrent:
            # Add additional time dimension if recurrent networks are used
            current_state = [tf.expand_dims(state, axis=1) for state in current_state]
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
        self.episodic_intrinsic_reward = self.alpha * (self.beta - similarity_score)

        # States must only be added to episodic memory if intrinsic reward is large enough/the state embedding is
        # relevant enough for the exploration process.
        if self.episodic_intrinsic_reward > self.novelty_threshold:
            self.episodic_memory.append(state_embedding)

        return self.episodic_intrinsic_reward

    def get_logs(self):
        if self.index == 0:
            return {"Exploration/EpisodicLoss": self.loss,
                    "Exploration/IntrinsicReward": self.episodic_intrinsic_reward}
        else:
            return {"Exploration/EpisodicLoss": self.loss}

    def reset(self):
        """Empty episodic memory"""
        self.episodic_memory.clear()
        return

    def prevent_checkpoint(self):
        return False
