#!/usr/bin/env python

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from ..misc.network_constructor import construct_network
from tensorflow.keras.models import clone_model
from .agent_blueprint import Actor, Learner
from datetime import datetime
import os
import ray
import tensorflow as tf

@ray.remote
class DQNActor(Actor):
    def __init__(self, port: int, mode: str,
                 interface: str,
                 preprocessing_algorithm: str,
                 preprocessing_path: str,
                 exploration_algorithm: str,
                 environment_path: str = "",
                 device: str = '/cpu:0'):
        super().__init__(port, mode, interface, preprocessing_algorithm, preprocessing_path,
                         exploration_algorithm, environment_path, device)

    def act(self, states, agent_ids=None,  mode="training"):
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
                actions = tf.expand_dims(tf.argmax(action_values, axis=1), axis=1)
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
                                                                         self.action_shape, self.sequence_length)
        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_replay_batch(samples, self.observation_shapes, self.action_shape)
        if np.any(action_batch is None):
            return None

        with tf.device(self.device):
            row_array = np.arange(len(samples))
            target_prediction = self.critic_prediction_network(next_state_batch)
            target_batch = reward_batch + \
                (self.gamma**self.n_steps) * tf.maximum(target_prediction, axis=1) * (1-done_batch)

            # Set the Q value of the chosen action to the target.
            q_batch = self.model(state_batch)
            q_batch[row_array, action_batch.astype(int)] = target_batch

            # Train the network on the training batch.
            sample_errors = np.abs(q_batch - self.critic_prediction_network(state_batch))
            if self.recurrent:
                eta = 0.9
                sample_errors = eta*np.max(sample_errors, axis=1) + (1-eta)*np.mean(sample_errors, axis=1)

        return sample_errors

    def update_actor_network(self, network_weights):
        self.critic_network.set_weights(network_weights)
        if self.recurrent:
            self.critic_prediction_network.set_weights(network_weights)
        self.network_update_requested = False
        self.steps_taken_since_network_update = 0

    def build_network(self, network_parameters, environment_parameters, idx):
        # Critic
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None]
        network_parameters[0]['TargetNetwork'] = False
        network_parameters[0]['NetworkType'] = 'ModelCopy{}'.format(idx)
        # Recurrent Parameters
        network_parameters[0]['Recurrent'] = self.recurrent
        network_parameters[0]['ReturnSequences'] = False
        network_parameters[0]['ReturnStates'] = True
        network_parameters[0]['Stateful'] = False
        network_parameters[0]['BatchSize'] = None  # self.agent_number

        # Actor for Error Prediction
        network_parameters[1] = network_parameters[0].copy()
        network_parameters[1]['NetworkType'] = 'CriticPredictionNetwork{}'.format(idx)
        network_parameters[1]['ReturnSequences'] = True
        network_parameters[1]['ReturnStates'] = False
        network_parameters[1]['Stateful'] = False
        network_parameters[1]['BatchSize'] = None

        # Build
        with tf.device(self.device):
            self.critic_network = construct_network(network_parameters[0])
            self.critic_prediction_network = construct_network(network_parameters[1])
        return True


@ray.remote(num_gpus=1)
class DQNLearner(Learner):
    # region ParameterSpace
    ActionType = ['DISCRETE']
    NetworkTypes = ['QNetwork']
    # endregion

    def __init__(self, mode, trainer_configuration, environment_configuration, model_path=None):
        super().__init__(trainer_configuration, environment_configuration)
        # Networks
        self.model, self.model_target = None, None

        # Double Learning
        self.double_learning = trainer_configuration.get('DoubleLearning')

        # Learning Parameters
        self.learning_rate = trainer_configuration.get('LearningRate')

        # Construct or load the required neural networks based on the trainer configuration and environment information
        if mode == 'training':
            # Network Construction
            self.build_network(network_parameters, environment_configuration)
            # Load Pretrained Models
            if model_path:
                self.load_checkpoint(model_path)

            # Compile Networks
            self.model.compile(optimizer=Adam(learning_rate=trainer_configuration.get('LearningRate'),
                                              clipvalue=self.clip_grad), loss="mse")

        # Load trained Models
        elif mode == 'testing':
            assert model_path, "No model path entered."
            self.load_checkpoint(model_path)

    def get_actor_network_weights(self):
        return [self.model.get_weights()]

    def build_network(self, network_parameters, environment_parameters):
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None]
        network_parameters[0]['TargetNetwork'] = True
        network_parameters[0]['NetworkType'] = self.NetworkTypes[0]
        # Recurrent Parameters
        network_parameters[0]['Recurrent'] = self.recurrent
        network_parameters[0]['ReturnSequences'] = True
        network_parameters[0]['Stateful'] = True
        network_parameters[0]['ReturnStates'] = False
        network_parameters[0]['BatchSize'] = self.batch_size

        # Build
        self.model, self.model_target = construct_network(network_parameters[0])

    def learn(self, replay_batch):
        # region --- REPLAY BATCH PREPROCESSING
        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)

            # Reset hidden LSTM states for all networks
            for net in [self.model, self.model_target]:
                for layer in net.layers:
                    if "lstm" in layer.name:
                        layer.reset_states()

            if np.any(action_batch is None):
                return None, None, self.training_step

            # Create Burn In Batches
            if self.burn_in:
                # Separate batch into 2 sequences each
                state_batch, action_batch, reward_batch, next_state_batch, done_batch, \
                    state_batch_burn, action_batch_burn, reward_batch_burn, next_state_batch_burn, done_batch_burn = \
                        Learner.separate_burn_in_batch(state_batch, action_batch, reward_batch,
                                                       next_state_batch, done_batch, self.burn_in)

        else:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)
        if np.any(action_batch is None):
            return None, None, self.training_step
        # endregion

        row_array = np.arange(len(replay_batch))
        # Burn in Critic Target Networks and Actor Network
        if self.recurrent and self.burn_in:
            self.target_model(next_state_batch_burn)

        # If the state is not terminal:
        # t = 𝑟 + 𝛾 * 𝑚𝑎𝑥_𝑎′ 𝑄̂(𝑠′,𝑎′) else t = r
        target_prediction = self.target_model(next_state_batch)
        if self.double_learning:
            if self.recurrent and self.burn_in:
                self.model(next_state_batch_burn)
            model_prediction_argmax = tf.argmax(self.model(next_state_batch), axis=1)
            target_batch = reward_batch + \
                (self.gamma**self.n_steps) * target_prediction[row_array, model_prediction_argmax] * (1-done_batch)
        else:
            target_batch = reward_batch + \
                (self.gamma**self.n_steps) * tf.maximum(target_prediction, axis=1) * (1-done_batch)

        if self.recurrent:
            for net in [self.model]:
                for layer in net.layers:
                    if "lstm" in layer.name:
                        layer.reset_states()
            if self.burn_in:
                self.model(state_batch_burn)
        # Set the Q value of the chosen action to the target.
        q_batch = self.model(state_batch)
        q_batch[row_array, action_batch.astype(int)] = target_batch

        # Train the network on the training batch.
        sample_errors = np.abs(q_batch - self.critic_network(state_batch))

        if self.recurrent:
            for net in [self.model]:
                for layer in net.layers:
                    if "lstm" in layer.name:
                        layer.reset_states()

        value_loss = self.model.train_on_batch(state_batch, q_batch)

        # Update target network weights
        self.training_step += 1
        self.sync_models()
        return {'Losses/Loss': value_loss}, sample_errors, self.training_step

    def sync_models(self):
        if self.sync_mode == "hard_sync":
            if not self.training_step % self.sync_steps and self.training_step > 0:
                self.model_target.set_weights(self.model.get_weights())
        elif self.sync_mode == "soft_sync":
            self.model_target.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                           for weights, target_weights in zip(self.model.get_weights(),
                                                                              self.model_target.get_weights())])
        else:
            raise ValueError("Sync mode unknown.")

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            self.model = load_model(path)
        elif os.path.isdir(path):
            file_names = [f for f in os.listdir(path) if f.endswith(".h5")]
            for file_name in file_names:
                if "DQN_Model" in file_name:
                    self.model = load_model(os.path.join(path, file_name))
                    self.model_target = clone_model(self.model)
                    self.model_target.set_weights(self.model.get_weights())
            if not self.model:
                raise FileNotFoundError("Could not find all necessary model files.")
        else:
            raise NotADirectoryError("Could not find directory or file for loading models.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        self.model.save(
            os.path.join(path, "DQN_Model_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))

    @staticmethod
    def get_config():
        config_dict = DQNLearner.__dict__
        return Learner.get_config(config_dict)
