import tensorflow as tf
import numpy as np
import time
from tensorflow.keras.losses import MeanSquaredError, CategoricalCrossentropy, BinaryCrossentropy
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow as tf
import itertools
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
import time
from collections import deque
from tensorflow.keras.utils import plot_model

np.random.seed(42)
tf.random.set_seed(42)


class Mooodel():
    def __init__(self):
        self.observation_shapes = [(35,)]
        self.action_shape = (5,)
        self.feature_space_size = 32
        self.recurrent = True

        # Loss metrics
        self.cce = CategoricalCrossentropy()
        self.bce = BinaryCrossentropy()
        self.optimizer = Adam(1e-4)
        self.loss = 0
        self.intrinsic_reward = 0

        # region Episodic Curiosity module parameters
        self.k = 5
        self.novelty_threshold = 0
        self.beta = 1
        self.alpha = 1
        self.gamma = 5
        self.episodic_memory = deque(maxlen=600)
        self.reset_episodic_memory = True  # exploration_parameters["ResetEpisodicMemory"]

        # Override observation shapes
        modified_observation_shapes = []
        for obs_shape in self.observation_shapes:
            modified_observation_shapes.append(obs_shape)
        '''modified_observation_shapes.append((1,))
        modified_observation_shapes.append((1,))'''
        self.observation_shapes = modified_observation_shapes

        # region Feature Extractor
        self.action_space = "DISCRETE"
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
        self.feature_extractor = Model(feature_input, x, name="ENM Feature Extractor")
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
        self.embedding_classifier = Model([current_state_features, next_state_features], x, name="ENM Classifier")
        # endregion

        # region Model compilation and plotting
        if self.action_space == "DISCRETE":
            self.feature_extractor.compile(loss=self.cce, optimizer=self.optimizer)
            self.embedding_classifier.compile(loss=self.cce, optimizer=self.optimizer)
        elif self.action_space == "CONTINUOUS":
            self.feature_extractor.compile(loss=self.mse, optimizer=self.optimizer)
            self.embedding_classifier.compile(loss=self.mse, optimizer=self.optimizer)

        # Model plots
        try:
            plot_model(self.feature_extractor, "plots/ENM_FeatureExtractor.png", show_shapes=True)
            plot_model(self.embedding_classifier, "plots/ENM_EmbeddingClassifier.png", show_shapes=True)
        except ImportError:
            print("Could not create model plots for ENM.")

        # Summaries
        self.feature_extractor.summary()
        self.embedding_classifier.summary()
        # endregion

        ''' # region Embedding Network
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
        self.feature_extractor = Model(feature_input, x, name="ECR Feature Extractor")
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
        self.comparator_network = Model([current_state_features, next_state_features], x, name="ECR Comparator")
        # endregion

        # region ALTERNATIVEComparator
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
        self.sigmoid_comparator_network = Model([current_state_features, next_state_features], x, name="ECR Comparator 2")

        # region Model compilation and plotting
        # comparator_network.compile(loss=self.cce, optimizer=self.optimizer)
        self.feature_extractor.compile(loss=self.bce, optimizer=self.optimizer)
        self.comparator_network.compile(loss=self.bce, optimizer=self.optimizer)
        self.sigmoid_comparator_network.compile(loss=self.bce, optimizer=self.optimizer)

        # Model plots
        try:
            plot_model(self.feature_extractor, "plots/ECR_FeatureExtractor.png", show_shapes=True)
            plot_model(self.comparator_network, "plots/ECR_Comparator.png", show_shapes=True)
        except ImportError:
            print("Could not create model plots for ECR.")

        # Summaries
        self.feature_extractor.summary()
        self.comparator_network.summary()
        self.sigmoid_comparator_network.summary()'''

if __name__ == '__main__':
    mdl = Mooodel()
    time.sleep(3)
    '''for a in range(100):
        mdl.episodic_memory.append(np.random.random((1, 32)))
    state_embedding = np.random.random((1, 32))
    state_embedding_array = np.empty([mdl.episodic_memory.__len__(), state_embedding.shape[0], state_embedding.shape[1]])
    state_embedding_array[:] = state_embedding
    reachability_buffer = mdl.comparator_network([state_embedding_array, np.array(mdl.episodic_memory)])
    reachability_buffer = reachability_buffer[:, :, 1]
    y_true = tf.convert_to_tensor(np.ones((32, 7)))'''

    state_input_size = 35
    x1 = np.random.random((32, 15, state_input_size))
    x2 = np.random.random((32, 15, state_input_size))
    action_batch = np.random.random_integers(0,4,(32, 15, 5))
    action_batch = action_batch[:, :, 0]
    # Calculate Loss
    with tf.GradientTape() as tape:
        # Calculate features of current and next state
        state_features = mdl.feature_extractor(x1)
        next_state_features = mdl.feature_extractor(x2)

        # Predict actions based on extracted features
        action_prediction = mdl.embedding_classifier([state_features, next_state_features])

        # Calculate inverse loss
        if mdl.action_space == "DISCRETE":
            # Encode true action as one hot vector encoding
            true_actions_one_hot = tf.one_hot(action_batch, mdl.action_shape[0])

            # Compute Loss via Categorical Cross Entropy
            mdl.loss = mdl.cce(true_actions_one_hot, action_prediction)

        elif mdl.action_space == "CONTINUOUS":
            # Compute Loss via Mean Squared Error
            mdl.loss = mdl.mse(action_batch, action_prediction)

    # Calculate Gradients
    grad = tape.gradient(mdl.loss, [mdl.embedding_classifier.trainable_weights,
                                     mdl.feature_extractor.trainable_weights])
    # Apply Gradients to all models
    mdl.optimizer.apply_gradients(zip(grad[0], mdl.embedding_classifier.trainable_weights))
    mdl.optimizer.apply_gradients(zip(grad[1], mdl.feature_extractor.trainable_weights))

    '''with tf.GradientTape() as tape:
        # Calculate features
        state_features1 = mdl.feature_extractor(x1)
        state_features2 = mdl.feature_extractor(x2)

        # Calculate reachability between observation pairs
        y_pred = mdl.sigmoid_comparator_network([state_features1, state_features2])

        y_true_batch = np.random.random((32, 15, 1))

        # Calculate Binary Cross-Entropy Loss
        mdl.loss = mdl.bce(y_true_batch, y_pred)

    # Calculate Gradients
    grad = tape.gradient(mdl.loss, [mdl.sigmoid_comparator_network.trainable_weights,
                                    mdl.feature_extractor.trainable_weights])

    # Apply Gradients to all models
    mdl.optimizer.apply_gradients(zip(grad[0], mdl.sigmoid_comparator_network.trainable_weights))
    mdl.optimizer.apply_gradients(zip(grad[1], mdl.feature_extractor.trainable_weights))'''

    '''
    #y_pred = y_pred[:, :, 1]
    with tf.GradientTape() as tape:
        y_pred = mdl.comparator_network([x1_embedding, x2_embedding])
        y_pred_sig = mdl.sigmoid_comparator_network([x1_embedding, x2_embedding])
        y_true = np.random.random((32, 7, 2))
        loss_bce = mdl.bce(y_true, y_pred)
        loss_cce = mdl.cce(y_true, y_pred)
        loss_bce_sig = mdl.bce(y_true, y_pred_sig[:, :, 0])'''

    print('debug')
