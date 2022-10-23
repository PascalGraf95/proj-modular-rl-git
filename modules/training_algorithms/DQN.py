#!/usr/bin/env python

import numpy as np
from tensorflow import keras
from keras.models import load_model
from keras import Model
from keras.optimizers import Adam
from ..misc.network_constructor import construct_network
from keras.models import clone_model
from .agent_blueprint import Actor, Learner
from datetime import datetime
import os
import ray
import tensorflow as tf


@ray.remote
class DQNActor(Actor):
    def __init__(self, idx: int, port: int, mode: str,
                 interface: str,
                 preprocessing_algorithm: str,
                 preprocessing_path: str,
                 exploration_algorithm: str,
                 environment_path: str = "",
                 demonstration_path: str = "",
                 device: str = '/cpu:0'):
        super().__init__(idx, port, mode, interface, preprocessing_algorithm, preprocessing_path,
                         exploration_algorithm, environment_path, demonstration_path, device)

    def act(self, states, agent_ids=None, mode="training", clone=False):
        # Check if any agent in the environment is not in a terminal state
        active_agent_number = np.shape(states[0])[0]
        if not active_agent_number:
            return Learner.get_dummy_action(active_agent_number, self.action_shape, self.action_type)
        with tf.device(self.device):
            if self.recurrent:
                # Set the initial LSTM states correctly according to the number of active agents
                self.set_lstm_states(agent_ids)
                # In case of a recurrent network, the state input needs an additional time dimension
                states = [tf.expand_dims(state, axis=1) for state in states]
                action_values, hidden_state, cell_state = self.critic_network(states)
                actions = tf.expand_dims(tf.argmax(action_values[0], axis=1), axis=1)
                self.update_lstm_states(agent_ids, [hidden_state.numpy(), cell_state.numpy()])
            else:
                action_values = self.critic_network(states)
                actions = tf.expand_dims(tf.argmax(action_values, axis=1), axis=1)
        return actions.numpy()

    def get_sample_errors(self, samples):
        if not samples:
            return None

        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(samples, self.observation_shapes,
                                                                         1, self.sequence_length)

        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(samples, self.observation_shapes, 1)
        if np.any(action_batch is None):
            return None

        with tf.device(self.device):
            batch_array = np.arange(len(samples))

            if self.recurrent:
                target_prediction = self.critic_prediction_network(next_state_batch)
                y = self.critic_prediction_network(state_batch).numpy()
            else:
                target_prediction = self.critic_network(next_state_batch)
                y = self.critic_network(state_batch).numpy()

            # With additional network inputs, which is the case when using Agent57-concepts, the state batch contains an
            # idx giving the information which exploration policy was used during acting. This exploration policy
            # contains the parameters gamma (n-step learning) and beta (intrinsic reward scaling factor).
            if self.recurrent and self.policy_feedback:
                self.gamma = np.empty((np.shape(state_batch[-1])[0], np.shape(state_batch[-1])[1],
                                       self.critic_prediction_network.output_shape[-1]))
                # Get gammas through saved exploration policy indices within the sequences
                for idx, sequence in enumerate(state_batch[-1]):
                    self.gamma[idx][:] = self.exploration_degree[int(sequence[0][0])]['gamma']

            target_batch = reward_batch + np.multiply(np.multiply((self.gamma ** self.n_steps),
                                                                  np.max(target_prediction, axis=1, keepdims=True)),
                                                      (1 - done_batch))

            if self.recurrent:
                time_step_array = np.arange(self.sequence_length)
                mesh_x, mesh_y = np.meshgrid(time_step_array, batch_array)
                y[mesh_y, mesh_x, action_batch[:, :, 0].astype(int)] = target_batch[:, :, 0]
            else:
                y[batch_array, action_batch[:, 0].astype(int)] = target_batch[:, 0]

            # Train the network on the training batch.
            if self.recurrent:
                sample_errors = np.sum(np.abs(y - self.critic_prediction_network(state_batch)), axis=1)
            else:
                sample_errors = np.sum(np.abs(y - self.critic_network(state_batch)), axis=1)
            if self.recurrent:
                eta = 0.9
                sample_errors = eta * np.max(sample_errors, axis=1) + (1 - eta) * np.mean(sample_errors, axis=1)
        return sample_errors

    def update_actor_network(self, network_weights, total_episodes=0):
        if not len(network_weights):
            return
        self.critic_network.set_weights(network_weights[0])
        if self.recurrent:
            self.critic_prediction_network.set_weights(network_weights[0])
        self.network_update_requested = False
        self.steps_taken_since_network_update = 0

    def build_network(self, network_settings, environment_parameters):
        # Create a list of dictionaries with 2 entries, one for each network
        network_parameters = [{}, {}]
        # region  --- Critic ---
        # - Network Name -
        network_parameters[0]['NetworkName'] = "DQN_ModelCopy{}".format(self.index)
        # - Network Architecture-
        network_parameters[0]['VectorNetworkArchitecture'] = network_settings["VectorNetworkArchitecture"]
        network_parameters[0]['VisualNetworkArchitecture'] = network_settings["VisualNetworkArchitecture"]
        network_parameters[0]['Filters'] = network_settings["Filters"]
        network_parameters[0]['Units'] = network_settings["Units"]
        network_parameters[0]['TargetNetwork'] = False
        network_parameters[0]['DuelingNetworks'] = network_settings["DuelingNetworks"]
        network_parameters[0]['NoisyNetworks'] = network_settings["NoisyNetworks"]
        # - Input / Output / Initialization -
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None]
        # - Recurrent Parameters -
        network_parameters[0]['Recurrent'] = self.recurrent
        network_parameters[0]['ReturnSequences'] = False
        network_parameters[0]['Stateful'] = False
        network_parameters[0]['BatchSize'] = None
        network_parameters[0]['ReturnStates'] = True
        # endregion

        # region --- Error Prediction Critic for PER ---
        # The error prediction network is needed to calculate initial priorities for the prioritized experience replay
        # buffer.
        network_parameters[1] = network_parameters[0].copy()
        network_parameters[1]['NetworkName'] = 'DQN_ErrorPredictionModelCopy{}'.format(self.index)
        network_parameters[1]['ReturnSequences'] = True
        network_parameters[1]['ReturnStates'] = False
        network_parameters[1]['Stateful'] = False
        network_parameters[1]['BatchSize'] = None
        # endregion

        # region --- Build ---
        with tf.device(self.device):
            self.critic_network = construct_network(network_parameters[0], plot_network_model=(self.index == 0))
            if self.recurrent:
                self.critic_prediction_network = construct_network(network_parameters[1],
                                                                   plot_network_model=(self.index == 0))
                # In case of recurrent neural networks, the lstm layers need to be accessible so that the hidden and
                # cell states can be modified manually.
                self.get_lstm_layers()
        # endregion
        return True


@ray.remote
class DQNLearner(Learner):
    # region ParameterSpace
    ActionType = ['DISCRETE']
    NetworkTypes = ['QNetwork']

    # endregion

    def __init__(self, mode, trainer_configuration, environment_configuration, model_path=None, clone_model_path=None):
        super().__init__(trainer_configuration, environment_configuration, model_path, clone_model_path)
        # Networks
        self.critic: keras.Model
        self.critic_target: keras.Model

        # Double Learning
        self.double_learning = trainer_configuration.get('DoubleLearning')

        # Learning Parameters
        self.learning_rate = trainer_configuration.get('LearningRate')

        # Construct or load the required neural networks based on the trainer configuration and environment information
        if mode == 'training':
            # Network Construction
            self.build_network(trainer_configuration["NetworkParameters"], environment_configuration)
            # Try to load pretrained models if provided. Otherwise, this method does nothing.
            model_key = self.get_model_key_from_dictionary(self.model_dictionary, mode="latest")
            if model_key:
                self.load_checkpoint_from_path_list(self.model_dictionary[model_key]['ModelPaths'], clone=False)
            # TODO: Implement Clone model and self-play

            # Compile Networks
            self.critic.compile(optimizer=Adam(learning_rate=trainer_configuration.get('LearningRate'),
                                               clipvalue=self.clip_grad), loss=self.burn_in_mse_loss)

        # Load trained Models
        elif mode == 'testing':
            assert model_path, "No model path entered."
            # Try to load pretrained models if provided. Otherwise, this method does nothing.
            model_key = self.get_model_key_from_dictionary(self.model_dictionary, mode="latest")
            if model_key:
                self.load_checkpoint_from_path_list(self.model_dictionary[model_key]['ModelPaths'], clone=False)

    def get_actor_network_weights(self, update_requested):
        return [self.critic.get_weights()]

    def build_network(self, network_settings, environment_parameters):
        # Create a list of dictionaries with 1 entry, one for each network
        network_parameters = [{}]

        # region --- Critic ---
        # - Network Name -
        network_parameters[0]['NetworkName'] = "DQN_" + self.NetworkTypes[0]
        # - Network Architecture-
        network_parameters[0]['VectorNetworkArchitecture'] = network_settings["VectorNetworkArchitecture"]
        network_parameters[0]['VisualNetworkArchitecture'] = network_settings["VisualNetworkArchitecture"]
        network_parameters[0]['Filters'] = network_settings["Filters"]
        network_parameters[0]['Units'] = network_settings["Units"]
        network_parameters[0]['TargetNetwork'] = True
        network_parameters[0]['DuelingNetworks'] = network_settings["DuelingNetworks"]
        network_parameters[0]['NoisyNetworks'] = network_settings["NoisyNetworks"]
        # - Input / Output / Initialization -
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None]
        # For image-based environments, the critic receives a mixture of images and vectors as input. If the following
        # option is true, the vector inputs will be repeated and stacked into the form of an image.
        network_parameters[0]['Vec2Img'] = False

        # - Recurrent Parameters -
        network_parameters[0]['Recurrent'] = self.recurrent
        # For loss calculation the recurrent critic needs to return one output per sample in the training sequence.
        network_parameters[0]['ReturnSequences'] = True
        # The critic no longer needs to be stateful due to new training process. This means the hidden and cell states
        # are reset after every prediction. Batch size thus also does not need to be predefined.
        network_parameters[0]['Stateful'] = False
        network_parameters[0]['BatchSize'] = None
        # The hidden network states are not relevant.
        network_parameters[0]['ReturnStates'] = False
        # endregion

        # region --- Building ---
        # Build the networks from the network parameters
        self.critic, self.critic_target = construct_network(network_parameters[0], plot_network_model=True)
        # endregion

    @ray.method(num_returns=3)
    def learn(self, replay_batch):
        if not replay_batch:
            return None, None, self.training_step

        # region --- REPLAY BATCH PREPROCESSING ---
        batch_array = np.arange(len(replay_batch))

        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         1, self.sequence_length)

            time_step_array = np.arange(self.sequence_length)
            mesh_x, mesh_y = np.meshgrid(time_step_array, batch_array)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, 1)
        if np.any(action_batch is None):
            return None, None, self.training_step
        # endregion

        # With additional network inputs, which is the case when using Agent57-concepts, the state batch contains an idx
        # giving the information which exploration policy was used during acting. This exploration policy contains the
        # parameters gamma (n-step learning) and beta (intrinsic reward scaling factor).
        if self.policy_feedback:
            self.gamma = np.empty((np.shape(state_batch[-1])[0], np.shape(state_batch[-1])[1],
                                   self.critic.output_shape[-1]))
            # Get gammas through saved exploration policy indices within the sequences
            for idx, sequence in enumerate(state_batch[-1]):
                self.gamma[idx][:] = self.exploration_degree[int(sequence[0][0])]['gamma']

        # If the state is not terminal:
        # t = 𝑟 + 𝛾 * 𝑚𝑎𝑥_𝑎′ 𝑄̂(𝑠′,𝑎′) else t = r
        # target_prediction: (batch_size, time_steps, action_size) or (batch_size, action_size)
        target_prediction = self.critic_target(next_state_batch).numpy()
        if self.double_learning:
            # model_prediction_argmax : (batch_size,time_steps) or (batch_size,)
            model_prediction_argmax = np.argmax(self.critic(next_state_batch).numpy(), axis=-1)
            if self.recurrent:
                # target_batch: (32, 10, 1) or (32, 1)
                target_batch = reward_batch + np.multiply(np.multiply((self.gamma ** self.n_steps),
                               np.expand_dims(target_prediction[mesh_y, mesh_x, model_prediction_argmax], axis=-1)),
                                                          (1 - done_batch))
            else:
                target_batch = reward_batch + np.multiply(np.multiply((self.gamma ** self.n_steps),
                               np.expand_dims(target_prediction[batch_array, model_prediction_argmax], axis=-1)),
                                                          (1 - done_batch))
        else:
            target_batch = reward_batch + np.multiply(np.multiply((self.gamma ** self.n_steps),
                                                                  tf.reduce_max(target_prediction, axis=-1,
                                                                                keepdims=True)), (1 - done_batch))

        # Set the Q value of the chosen action to the target.
        # y: (32, 10, 5) or (32, 5)
        y = self.critic(state_batch).numpy()
        # action_batch: (32, 10, 1) or (32, 1)
        if self.recurrent:
            y[mesh_y, mesh_x, action_batch[:, :, 0].astype(int)] = target_batch[:, :, 0]
        else:
            y[batch_array, action_batch[:, 0].astype(int)] = target_batch[:, 0]

        # Calculate sample errors
        sample_errors = np.abs(y - self.critic(state_batch))
        if self.recurrent:
            eta = 0.9
            sample_errors = eta * np.max(sample_errors[:, self.burn_in:], axis=1) + \
                            (1 - eta) * np.mean(sample_errors[:, self.burn_in:], axis=1)
        sample_errors = np.sum(sample_errors, axis=1)

        # Train the network on the training batch.
        value_loss = self.critic.train_on_batch(state_batch, y)

        # Update target network weights
        self.training_step += 1
        self.sync_models()
        return {'Losses/Loss': value_loss}, sample_errors, self.training_step

    def boost_exploration(self):
        pass

    def sync_models(self):
        if self.sync_mode == "hard_sync":
            if not self.training_step % self.sync_steps and self.training_step > 0:
                self.critic_target.set_weights(self.critic.get_weights())
        elif self.sync_mode == "soft_sync":
            self.critic_target.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                            for weights, target_weights in zip(self.critic.get_weights(),
                                                                               self.critic_target.get_weights())])
        else:
            raise ValueError("Sync mode unknown.")

    def load_checkpoint_from_path_list(self, model_paths, clone=False):
        if not clone:
            for file_path in model_paths:
                if "DQN_Critic" in file_path:
                    self.critic = load_model(file_path)
                if not self.critic:
                    raise FileNotFoundError("Could not find all necessary model files.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False,
                        checkpoint_condition=True):
        if not checkpoint_condition:
            return
        self.critic.save(
            os.path.join(path, "DQN_Critic_Step{:06d}_Reward{:.2f}".format(training_step, running_average_reward)))

    @staticmethod
    def get_config():
        config_dict = DQNLearner.__dict__
        return Learner.get_config(config_dict)
