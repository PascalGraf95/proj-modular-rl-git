import numpy as np
from tensorflow import keras
from modules.exploration_algorithms.exploration_algorithm_blueprint import ExplorationAlgorithm
from keras.losses import MeanSquaredError, CategoricalCrossentropy
from keras.optimizers import Adam, SGD
import tensorflow as tf
from modules.misc.utility import modify_observation_shapes
from modules.training_algorithms.agent_blueprint import Learner
import itertools
from keras import Input, Model
from keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
import time
from collections import deque
from keras.utils import plot_model


class NeverGiveUp(ExplorationAlgorithm):
    """
    Basic implementation of Agent57's reward generator NeverGiveUp (NGU)
    The computation of intrinsic episodic rewards is done for each actor and after every environment step (see act()).

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
        # However, MSE is also used within the life-long module.
        if self.action_space == "DISCRETE":
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
        self.episodic_novelty_module_built = False
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
        self.optimizer_episodic = Adam(learning_rate=0.00005)  # fixed learning rate for episodic parts (optional)
        # self.optimizer_episodic = Adam(exploration_parameters["LearningRate"])
        self.optimizer_lifelong = Adam(parameters["LearningRate"])
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

        # region - Modify observation shapes and construct neural networks -
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
        # to ICM. LNM utilizes also two network architectures, a prediction model which tries to predict the outputs
        # of a target model. Overall the LNM output is used as a modulator for up- or downscaling the ENM output.
        # Episodic Novelty Module Networks
        self.feature_extractor, self.embedding_classifier = self.build_network()
        self.episodic_novelty_module_built = True
        # Lifelong Novelty Module Networks
        self.prediction_model, self.target_model = self.build_network()
        # endregion

    def build_network(self):
        with tf.device(self.device):
            # region - Episodic Novelty Module -
            if not self.episodic_novelty_module_built:
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
                    plot_model(feature_extractor, "plots/ENM_Feature_Extractor.png", show_shapes=True)
                    plot_model(embedding_classifier, "plots/ENM_Embedding_Classifier.png", show_shapes=True)
                except ImportError:
                    print("Could not create model plots for ENM.")

                # Summaries
                feature_extractor.summary()
                embedding_classifier.summary()
                # endregion

                return feature_extractor, embedding_classifier
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

            # region - ENM -
            # Classifier and feature extractor are trained end-to-end on pairs of states and next states.
            # The feature extractor embeds both states, the classifier predicts which action lead to the respective
            # transition. This way it is ensured that the feature extractor only considers features and patterns that
            # play a role for determining the agent's action.
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
                    self.episodic_loss = self.mse(action_batch, action_prediction)

            # Calculate Gradients
            grad = tape.gradient(self.episodic_loss, [self.embedding_classifier.trainable_weights,
                                                      self.feature_extractor.trainable_weights])
            # Apply Gradients to all models
            self.optimizer_episodic.apply_gradients(zip(grad[0], self.embedding_classifier.trainable_weights))
            self.optimizer_episodic.apply_gradients(zip(grad[1], self.feature_extractor.trainable_weights))
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
        """
        Calculate intrinsically-based episodic reward through similarity comparison of current state embedding with
        the content of the episodic memory.
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
        # region - Feature embedding and distance calculation -
        if self.recurrent:
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
            episodic_intrinsic_reward = 0
        else:
            # Otherwise, the intrinsic episodic reward is calculated 1/similarity to encourage visiting
            # states with lower similarity (via extended observation space).
            episodic_intrinsic_reward = (1 / similarity)
        # endregion
        # endregion

        # region - Final intrinsic reward calculation -
        # Modulate episodic novelty module output by lifelong novelty module output
        #
        # NOTE: The following formula allows the algorithm to modulate the episodic_intrinsic_reward between
        # 1...alpha_max. This means that the exploration algorithm is at least driven by the value of the
        # episodic_intrinsic_reward. In some cases it would possibly make sense to let the lifelong module down-
        # modulate the episodic part to 0. For that, simply change "max(lifelong_intrinsic_reward, 1)" to
        # "max(lifelong_intrinsic_reward, 0)".
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
        self.mean_distances.clear()
        self.episodic_memory.clear()
        return

    def prevent_checkpoint(self):
        return False
