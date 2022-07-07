#!/usr/bin/env python
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from .agent_blueprint import Actor, Learner
from tensorflow.keras.models import load_model
from ..misc.network_constructor import construct_network
import tensorflow as tf
from tensorflow.keras.models import clone_model
from tensorflow.keras import losses
import tensorflow_probability as tfp
import os
import ray
import time

tfd = tfp.distributions
global AgentInterface


@ray.remote
class SACActor(Actor):
    def __init__(self, idx: int, port: int, mode: str,
                 interface: str,
                 preprocessing_algorithm: str,
                 preprocessing_path: str,
                 exploration_algorithm: str,
                 meta_learning_algorithm: str,
                 environment_path: str = "",
                 demonstration_path: str = "",
                 device: str = '/cpu:0'):
        super().__init__(idx, port, mode, interface, preprocessing_algorithm, preprocessing_path,
                         exploration_algorithm, meta_learning_algorithm, environment_path, demonstration_path, device)

    def act(self, states, agent_ids=None, mode="training", clone=False):
        # Check if any agent in the environment is not in a terminal state
        active_agent_number = len(agent_ids)
        if not active_agent_number:
            return Learner.get_dummy_action(active_agent_number, self.action_shape, self.action_type)
        with tf.device(self.device):
            if self.recurrent:
                # Set the initial LSTM states correctly according to the number of active agents
                self.set_lstm_states(agent_ids, clone=clone)
                # In case of a recurrent network, the state input needs an additional time dimension
                states = [tf.expand_dims(state, axis=1) for state in states]
                if clone:
                    (mean, log_std), hidden_state, cell_state = self.clone_actor_network(states)
                else:
                    (mean, log_std), hidden_state, cell_state = self.actor_network(states)
                # Update the LSTM states according to the latest network prediction
                self.update_lstm_states(agent_ids, [hidden_state.numpy(), cell_state.numpy()], clone=clone)
            else:
                if clone:
                    mean, log_std = self.clone_actor_network(states)
                else:
                    mean, log_std = self.actor_network(states)
            if mode == "training" and not clone:
                normal = tfd.Normal(mean, tf.exp(log_std))
                actions = tf.tanh(normal.sample())
            else:
                actions = tf.tanh(mean)
        return actions.numpy()

    def get_sample_errors(self, samples):
        """Calculates the prediction error for each state/sequence which corresponds to the initial priority in the
        prioritized experience replay buffer."""
        if not samples:
            return None

        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(samples, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(samples, self.observation_shapes, self.action_shape)
        if np.any(action_batch is None):
            return None

        with tf.device(self.device):
            if self.recurrent:
                mean, log_std = self.actor_prediction_network(next_state_batch)
            else:
                mean, log_std = self.actor_network(next_state_batch)
            normal = tfd.Normal(mean, tf.exp(log_std))
            next_actions = tf.tanh(normal.sample())
            # Critic Target Predictions
            critic_prediction = self.critic_network([*next_state_batch, next_actions])
            critic_target = critic_prediction * (1 - done_batch)

            # Use gamma given through exploration policy index (same value for entire sequence)
            if self.additional_network_inputs:
                # state_batch is sampled component-based for first index. The saved exploration policy index within the
                # observation is always the last observation component -> therefore state_batch[-1] is used to get it.
                # Get gammas through saved exploration policy indices within the sequences
                self.gamma = np.zeros(len(critic_target))
                for idx, sequence in enumerate(state_batch[-1]):
                    self.gamma[idx] = self.exploration_policies[int(sequence[0][0])]['gamma']
                # Add two pseudo dimensions to gamma for proper shape calculations regarding critic_target
                y = reward_batch + (self.gamma[:, None, None] ** self.n_steps) * critic_target
            else:
                y = reward_batch + (self.gamma ** self.n_steps) * critic_target

            # Train Both Critic Networks
            y = reward_batch + (self.gamma ** self.n_steps) * critic_target
            sample_errors = np.abs(y - self.critic_network([*state_batch, action_batch]))
            # In case of a recurrent agent the priority has to be averaged over each sequence according to the
            # formula in the paper
            if self.recurrent:
                eta = 0.9
                sample_errors = eta * np.max(sample_errors, axis=1) + (1 - eta) * np.mean(sample_errors, axis=1)
        return sample_errors

    def update_actor_network(self, network_weights, total_episodes=0):
        if not len(network_weights):
            return
        self.actor_network.set_weights(network_weights[0])
        self.critic_network.set_weights(network_weights[1])
        if self.recurrent:
            self.actor_prediction_network.set_weights(network_weights[0])
        if (self.is_clone_network_update_requested(total_episodes)) and self.behavior_clone_name:
            self.clone_actor_network.set_weights(network_weights[0])
            print("Clone Network has been updated")
            self.steps_taken_since_clone_network_update = 0
        self.steps_taken_since_network_update = 0

    def build_network(self, network_settings, environment_parameters):
        # Create a list of dictionaries with 4 entries, one for each network
        network_parameters = [{}, {}, {}, {}]
        # region --- Actor ---
        # - Network Name -
        network_parameters[0]['NetworkName'] = 'SAC_ActorCopy{}'.format(self.index)
        # - Network Architecture-
        network_parameters[0]['VectorNetworkArchitecture'] = network_settings["ActorVectorNetworkArchitecture"]
        network_parameters[0]['VisualNetworkArchitecture'] = network_settings["ActorVisualNetworkArchitecture"]
        network_parameters[0]['Filters'] = network_settings["Filters"]
        network_parameters[0]['Units'] = network_settings["Units"]
        # - Input / Output / Initialization -
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                           environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None, None]
        network_parameters[0]['KernelInitializer'] = "RandomUniform"
        # - Recurrent Parameters -
        network_parameters[0]['Recurrent'] = self.recurrent
        # For action calculation the recurrent actor only needs to return one output.
        network_parameters[0]['ReturnSequences'] = False
        # The network is not stateful because the batch size constantly changes. Therefor, the states need to be
        # returned and modified. The batch size thus is irrelevant.
        network_parameters[0]['Stateful'] = False
        network_parameters[0]['ReturnStates'] = True
        network_parameters[0]['BatchSize'] = None
        # endregion

        # region --- Error Prediction Actor for PER ---
        # The error prediction network is needed to calculate initial priorities for the prioritized experience replay
        # buffer.
        network_parameters[2] = network_parameters[0].copy()
        network_parameters[2]['NetworkName'] = 'SAC_ActorErrorPredictionCopy{}'.format(self.index)
        network_parameters[2]['ReturnSequences'] = True
        network_parameters[2]['ReturnStates'] = False
        network_parameters[2]['Stateful'] = False
        network_parameters[2]['BatchSize'] = None
        # endregion

        # region  --- Critic ---
        # The critic network is needed to calculate initial priorities for the prioritized experience replay
        # buffer.
        # - Network Name -
        network_parameters[1]['NetworkName'] = "SAC_CriticCopy{}".format(self.index)
        # - Network Architecture-
        network_parameters[1]['VectorNetworkArchitecture'] = network_settings["CriticVectorNetworkArchitecture"]
        network_parameters[1]['VisualNetworkArchitecture'] = network_settings["CriticVisualNetworkArchitecture"]
        network_parameters[1]['Filters'] = network_settings["Filters"]
        network_parameters[1]['Units'] = network_settings["Units"]
        network_parameters[1]['TargetNetwork'] = False
        # - Input / Output / Initialization -
        network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                          environment_parameters.get('ActionShape')]
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['KernelInitializer'] = "RandomUniform"
        # Recurrent Parameters
        network_parameters[1]['Recurrent'] = self.recurrent
        network_parameters[1]['ReturnSequences'] = True
        network_parameters[1]['Stateful'] = False
        network_parameters[1]['BatchSize'] = None
        network_parameters[1]['ReturnStates'] = False
        # endregion

        # region --- Build ---
        with tf.device(self.device):
            self.actor_network = construct_network(network_parameters[0], plot_network_model=True)
            self.critic_network = construct_network(network_parameters[1])
            if self.recurrent:
                self.actor_prediction_network = construct_network(network_parameters[2])
                # In case of recurrent neural networks, the lstm layers need to be accessible so that the hidden and
                # cell states can be modified manually.
                self.get_lstm_layers()
            # If there is a clone agent in the environment, instantiate another actor network for self-play.
            if self.behavior_clone_name:
                network_parameters.append(network_parameters[0].copy())
                network_parameters[3]['NetworkName'] = 'ActorCloneCopy{}'.format(self.index)
                self.clone_actor_network = construct_network(network_parameters[3])
        # endregion
        return True


@ray.remote
class SACLearner(Learner):
    ActionType = ['CONTINUOUS']
    NetworkTypes = ['Actor', 'Critic1', 'Critic2']

    def __init__(self, mode, trainer_configuration, environment_configuration, model_path=None):
        super().__init__(trainer_configuration, environment_configuration)

        # - Neural Networks -
        # The Soft Actor-Critic algorithm utilizes 5 neural networks. One actor and two critics with one target network
        # each. The actor takes in the current state and outputs an action vector which consists of a mean and standard
        # deviation for each action component. This is the only network needed for acting after training.
        # Each critic takes in the current state as well as an action and predicts its Q-Value.
        self.actor_network: keras.Model
        self.critic1: keras.Model
        self.critic_target1: keras.Model
        self.critic2: keras.Model
        self.critic_target2: keras.Model
        # A small parameter epsilon prevents math errors when log probabilities are 0.
        self.epsilon = 1.0e-6

        # - Optimizer -
        self.actor_optimizer: keras.optimizers.Optimizer
        self.alpha_optimizer: keras.optimizers.Optimizer

        # - Temperature Parameter Alpha -
        # The alpha parameter similar to the epsilon in epsilon-greedy promotes exploring the environment by keeping
        # the standard deviation for each action as high as possible while still performing the task. However, in
        # contrast to epsilon, alpha is a learnable parameter and adjusts automatically.
        self.log_alpha = tf.Variable(tf.ones(1) * trainer_configuration.get('LogAlpha'),
                                     constraint=lambda x: tf.clip_by_value(x, -20, 20), trainable=True)
        self.target_entropy = -tf.reduce_sum(tf.ones(self.action_shape))

        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=trainer_configuration.get('LearningRateActor'),
                                                        clipvalue=self.clip_grad)
        self.alpha = tf.exp(self.log_alpha).numpy()

        # Construct or load the required neural networks based on the trainer configuration and environment information
        if mode == 'training':
            # Network Construction
            self.build_network(trainer_configuration.get("NetworkParameters"), environment_configuration)
            # Load Pretrained Models
            if model_path:
                self.load_checkpoint(model_path)

            # Compile Networks
            self.critic1.compile(optimizer=Adam(learning_rate=trainer_configuration.get('LearningRateCritic'),
                                                clipvalue=self.clip_grad), loss=self.burn_in_mse_loss)
            self.critic2.compile(optimizer=Adam(learning_rate=trainer_configuration.get('LearningRateCritic'),
                                                clipvalue=self.clip_grad), loss=self.burn_in_mse_loss)
            self.actor_optimizer = Adam(learning_rate=trainer_configuration.get('LearningRateActor'),
                                        clipvalue=self.clip_grad)

        # Load trained Models
        elif mode == 'testing' or mode == 'fastTesting':
            assert model_path, "No model path entered."
            self.load_checkpoint(model_path)

    def get_actor_network_weights(self, update_requested):
        if not update_requested:
            return []
        return [self.actor_network.get_weights(), self.critic1.get_weights()]

    def build_network(self, network_settings, environment_parameters):
        # Create a list of dictionaries with 3 entries, one for each network
        network_parameters = [{}, {}, {}]
        # region --- Actor ---
        # - Network Name -
        network_parameters[0]['NetworkName'] = "SAC_" + self.NetworkTypes[0]
        # - Network Architecture-
        network_parameters[0]['VectorNetworkArchitecture'] = network_settings["ActorVectorNetworkArchitecture"]
        network_parameters[0]['VisualNetworkArchitecture'] = network_settings["ActorVisualNetworkArchitecture"]
        network_parameters[0]['Filters'] = network_settings["Filters"]
        network_parameters[0]['Units'] = network_settings["Units"]
        # - Input / Output / Initialization -
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                           environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None, None]
        network_parameters[0]['KernelInitializer'] = "RandomUniform"
        # - Recurrent Parameters -
        network_parameters[0]['Recurrent'] = self.recurrent
        # For loss calculation the recurrent actor needs to return one output per sample in the training sequence.
        network_parameters[0]['ReturnSequences'] = True
        # The actor no longer needs to be stateful due to new training process. This means the hidden and cell states
        # are reset after every prediction. Batch size thus also does not need to be predefined.
        network_parameters[0]['Stateful'] = False
        network_parameters[0]['BatchSize'] = None
        # The hidden network states are not relevant.
        network_parameters[0]['ReturnStates'] = False
        # endregion

        # region --- Critic1 ---
        # - Network Name -
        network_parameters[1]['NetworkName'] = "SAC_" + self.NetworkTypes[1]
        # - Network Architecture-
        network_parameters[1]['VectorNetworkArchitecture'] = network_settings["CriticVectorNetworkArchitecture"]
        network_parameters[1]['VisualNetworkArchitecture'] = network_settings["CriticVisualNetworkArchitecture"]
        network_parameters[1]['Filters'] = network_settings["Filters"]
        network_parameters[1]['Units'] = network_settings["Units"]
        network_parameters[1]['TargetNetwork'] = True
        # - Input / Output / Initialization -
        network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                          environment_parameters.get('ActionShape')]
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['KernelInitializer'] = "RandomUniform"
        # For image-based environments, the critic receives a mixture of images and vectors as input. If the following
        # option is true, the vector inputs will be repeated and stacked into the form of an image.
        network_parameters[1]['Vec2Img'] = False

        # - Recurrent Parameters -
        network_parameters[1]['Recurrent'] = self.recurrent
        # For loss calculation the recurrent critic needs to return one output per sample in the training sequence.
        network_parameters[1]['ReturnSequences'] = True
        # The critic no longer needs to be stateful due to new training process. This means the hidden and cell states
        # are reset after every prediction. Batch size thus also does not need to be predefined.
        network_parameters[1]['Stateful'] = False
        network_parameters[1]['BatchSize'] = None
        # The hidden network states are not relevant.
        network_parameters[1]['ReturnStates'] = False
        # endregion

        # region --- Critic2 ---
        # The second critic is an exact copy of the first one
        network_parameters[2] = network_parameters[1].copy()
        network_parameters[2]['NetworkName'] = "SAC_" + self.NetworkTypes[2]
        # endregion

        # region --- Building ---
        # Build the networks from the network parameters
        self.actor_network = construct_network(network_parameters[0], plot_network_model=True)
        self.critic1, self.critic_target1 = construct_network(network_parameters[1], plot_network_model=True)
        self.critic2, self.critic_target2 = construct_network(network_parameters[2])
        # endregion

    def forward(self, states):
        # Calculate the actors output and clip the logarithmic standard deviation values
        mean, log_std = self.actor_network(states)
        log_std = tf.clip_by_value(log_std, -20, 3)
        # Construct a normal function with mean and std and sample an action
        normal = tfd.Normal(mean, tf.exp(log_std))
        z = normal.sample()
        action = tf.tanh(z)

        # Calculate the logarithmic probability of z being sampled from the normal distribution.
        log_prob = normal.log_prob(z)
        log_prob_normalizer = tf.math.log(1 - action ** 2 + self.epsilon)
        log_prob -= log_prob_normalizer

        if self.recurrent:
            log_prob = tf.reduce_sum(log_prob, axis=2, keepdims=True)
        else:
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        return action, log_prob

    @ray.method(num_returns=3)
    def learn(self, replay_batch):
        if not replay_batch:
            return None, None, self.training_step

        # region --- REPLAY BATCH PREPROCESSING ---
        # In case of recurrent neural networks the batches have to be processed differently
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        if np.any(action_batch is None):
            return None, None, self.training_step
        # endregion

        # region --- CRITIC TRAINING ---
        next_actions, next_log_prob = self.forward(next_state_batch)

        # Critic Target Predictions
        critic_target_prediction1 = self.critic_target1([*next_state_batch, next_actions])
        critic_target_prediction2 = self.critic_target2([*next_state_batch, next_actions])
        critic_target_prediction = tf.minimum(critic_target_prediction1, critic_target_prediction2)

        # Possible Reward DeNormalization
        if self.reward_normalization:
            critic_target_prediction = self.inverse_value_function_rescaling(critic_target_prediction)

        # Training Target Calculation with standard TD-Error + Temperature Parameter
        critic_target = (critic_target_prediction - self.alpha * next_log_prob) * (1 - done_batch)

        # Use gamma given through exploration policy index (same value for entire sequence)
        if self.additional_network_inputs:
            # state_batch is sampled component-based for first index. The saved exploration policy index within the
            # observation is always the last observation component -> therefore state_batch[-1] is used to get it.
            # Get gammas through saved exploration policy indices within the sequences
            self.gamma = np.zeros(len(critic_target))
            for idx, sequence in enumerate(state_batch[-1]):
                self.gamma[idx] = self.exploration_policies[int(sequence[0][0])]['gamma']
            # Add two pseudo dimensions to gamma for proper shape calculations regarding critic_target
            y = reward_batch + (self.gamma[:, None, None] ** self.n_steps) * critic_target
        else:
            y = reward_batch + (self.gamma ** self.n_steps) * critic_target

        # Possible Reward Normalization
        if self.reward_normalization:
            y = self.value_function_rescaling(y)

        # Calculate Sample Errors to update priorities in Prioritized Experience Replay
        sample_errors = np.abs(y - self.critic1([*state_batch, action_batch]))
        if self.recurrent:
            eta = 0.9
            sample_errors = eta * np.max(sample_errors[:, self.burn_in:], axis=1) + \
                            (1 - eta) * np.mean(sample_errors[:, self.burn_in:], axis=1)
        # Calculate Critic 1 and 2 Loss, utilizes custom mse loss function defined in Trainer-class
        value_loss1 = self.critic1.train_on_batch([*state_batch, action_batch], y)
        value_loss2 = self.critic2.train_on_batch([*state_batch, action_batch], y)
        value_loss = (value_loss1 + value_loss2) / 2
        # endregion

        # region --- ACTOR TRAINING ---
        with tf.GradientTape() as tape:
            new_actions, log_prob = self.forward(state_batch)
            critic_prediction1 = self.critic1([*state_batch, new_actions])
            critic_prediction2 = self.critic2([*state_batch, new_actions])
            critic_prediction = tf.minimum(critic_prediction1, critic_prediction2)
            policy_loss = tf.reduce_mean(self.alpha * log_prob[:, self.burn_in:] - critic_prediction[:, self.burn_in:])

        actor_grads = tape.gradient(policy_loss, self.actor_network.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_variables))
        # endregion

        # region --- TEMPERATURE PARAMETER TRAINING ---
        with tf.GradientTape() as tape:
            alpha_loss = tf.reduce_mean(self.log_alpha * (-log_prob - self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        self.alpha = tf.exp(self.log_alpha).numpy()
        # endregion

        self.training_step += 1
        self.steps_since_actor_update += 1
        self.sync_models()

        return {'Losses/Loss': policy_loss + value_loss, 'Losses/PolicyLoss': policy_loss,
                'Losses/ValueLoss': value_loss, 'Losses/AlphaLoss': alpha_loss,
                'Losses/Alpha': tf.reduce_mean(self.alpha).numpy()}, sample_errors, self.training_step

    @staticmethod
    def value_function_rescaling(x, eps=1e-3):
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x

    @staticmethod
    def inverse_value_function_rescaling(h, eps=1e-3):
        return np.sign(h) * (((np.sqrt(1 + 4 * eps * (np.abs(h) + 1 + eps)) - 1) / (2 * eps)) - 1)

    def sync_models(self):
        if self.sync_mode == "hard_sync":
            if not self.training_step % self.sync_steps and self.training_step > 0:
                self.critic_target1.set_weights(self.critic1.get_weights())
                self.critic_target2.set_weights(self.critic2.get_weights())
        elif self.sync_mode == "soft_sync":
            self.critic_target1.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                             for weights, target_weights in zip(self.critic1.get_weights(),
                                                                                self.critic_target1.get_weights())])
            self.critic_target2.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                             for weights, target_weights in zip(self.critic2.get_weights(),
                                                                                self.critic_target2.get_weights())])
        else:
            raise ValueError("Sync mode unknown.")

    def load_checkpoint(self, path):
        if "Step" in path:
            self.actor_network = load_model(path)
        elif os.path.isdir(path):
            file_names = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
            for file_name in file_names:
                if "Critic1" in file_name:
                    self.critic1 = load_model(os.path.join(path, file_name), compile=False)
                    self.critic_target1 = clone_model(self.critic1)
                    self.critic_target1.set_weights(self.critic1.get_weights())
                elif "Critic2" in file_name:
                    self.critic2 = load_model(os.path.join(path, file_name), compile=False)
                    self.critic_target2 = clone_model(self.critic2)
                    self.critic_target2.set_weights(self.critic2.get_weights())
                elif "Actor" in file_name:
                    self.actor_network = load_model(os.path.join(path, file_name), compile=False)
            if not self.actor_network or not self.critic1 or not self.critic2:
                raise FileNotFoundError("Could not find all necessary model files.")
        else:
            raise NotADirectoryError("Could not find directory or file for loading models.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False,
                        checkpoint_condition=True):
        if not checkpoint_condition:
            return
        self.actor_network.save(
            os.path.join(path, "SAC_Actor_Step{}_Reward{:.2f}".format(training_step, running_average_reward)))
        if save_all_models:
            self.critic1.save(
                os.path.join(path, "SAC_Critic1_Step{}_Reward{:.2f}".format(training_step, running_average_reward)))
            self.critic2.save(
                os.path.join(path, "SAC_Critic2_Step{}_Reward{:.2f}".format(training_step, running_average_reward)))

    def boost_exploration(self):
        self.log_alpha = tf.Variable(tf.ones(1) * -0.7,
                                     constraint=lambda x: tf.clip_by_value(x, -10, 20), trainable=True)
        return True
