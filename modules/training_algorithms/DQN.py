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

    def act(self, states, mode="training"):
        # Check if any agent in the environment is not in a terminal state
        active_agent_number = np.shape(states[0])[0]
        if not active_agent_number:
            return Learner.get_dummy_action(active_agent_number, self.action_shape, self.action_type)
        with tf.device(self.device):
            action_values = self.critic_network(states)
            actions = tf.expand_dims(tf.argmax(action_values, axis=1), axis=1)
        return actions.numpy()

    def get_sample_errors(self, samples):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = self.get_training_batch_from_replay_batch(samples, self.observation_shapes, self.action_shape)

        row_array = np.arange(len(samples))
        # If the state is not terminal:
        # t = 𝑟 + 𝛾 * 𝑚𝑎𝑥_𝑎′ 𝑄̂(𝑠′,𝑎′) else t = r
        target_prediction = self.model.predict(next_state_batch)
        target_batch = reward_batch + \
            (self.gamma**self.n_steps) * tf.maximum(target_prediction, axis=1) * (1-done_batch)

        # Set the Q value of the chosen action to the target.
        q_batch = self.model.predict(state_batch)
        q_batch[row_array, action_batch.astype(int)] = target_batch

        # Train the network on the training batch.
        sample_errors = np.abs(q_batch - self.critic_network(state_batch))
        return sample_errors

    def update_actor_network(self, network_weights):
        self.critic_network.set_weights(network_weights)
        self.network_update_requested = False
        self.new_steps_taken = 0

    def build_network(self, network_parameters, environment_parameters, idx):
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None]
        network_parameters[0]['TargetNetwork'] = False
        network_parameters[0]['NetworkType'] = 'ModelCopy{}'.format(idx)

        # Build
        with tf.device(self.device):
            self.critic_network = construct_network(network_parameters[0])
        return True


@ray.remote(num_gpus=1)
class DQNLearner(Learner):
    # region ParameterSpace
    TrainingParameterSpace = Learner.TrainingParameterSpace.copy()
    DQNParameterSpace = {
        'LearningRate': float,
        'DoubleLearning': bool
    }
    TrainingParameterSpace = {**TrainingParameterSpace, **DQNParameterSpace}
    NetworkParameterSpace = [{
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
    }]
    ActionType = ['DISCRETE']
    NetworkTypes = ['QNetwork']
    Metrics = ['ValueLoss']
    # endregion

    def __init__(self, mode, trainer_configuration, environment_configuration, network_parameters, model_path=None):
        # Networks
        self.model, self.model_target = None, None

        # Double Learning
        self.double_learning = trainer_configuration.get('DoubleLearning')

        # Environment Configuration
        self.action_shape = environment_configuration.get('ActionShape')
        self.observation_shapes = environment_configuration.get('ObservationShapes')

        # Learning Parameters
        self.n_steps = trainer_configuration.get('NSteps')
        self.gamma = trainer_configuration.get('Gamma')
        self.sync_mode = trainer_configuration.get('SyncMode')
        self.sync_steps = trainer_configuration.get('SyncSteps')
        self.tau = trainer_configuration.get('Tau')
        self.clip_grad = trainer_configuration.get('ClipGrad')
        self.learning_rate = trainer_configuration.get('LearningRate')

        # Misc
        self.training_step = 0
        self.set_gpu_growth()  # Important step to avoid tensorflow OOM errors when running multiprocessing!

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
        return [self.actor_network.get_weights(), self.critic1.get_weights()]

    def build_network(self, network_parameters, environment_parameters):
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None]
        network_parameters[0]['TargetNetwork'] = True
        network_parameters[0]['NetworkType'] = self.NetworkTypes[0]

        # Build
        self.model, self.model_target = construct_network(network_parameters[0])

    def learn(self, replay_batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        row_array = np.arange(len(replay_batch))
        # If the state is not terminal:
        # t = 𝑟 + 𝛾 * 𝑚𝑎𝑥_𝑎′ 𝑄̂(𝑠′,𝑎′) else t = r
        target_prediction = self.target_model.predict(next_state_batch)
        if self.double_learning:
            model_prediction_argmax = tf.argmax(self.model(next_state_batch), axis=1)
            target_batch = reward_batch + \
                (self.gamma**self.n_steps) * target_prediction[row_array, model_prediction_argmax] * (1-done_batch)
        else:
            target_batch = reward_batch + \
                (self.gamma**self.n_steps) * tf.maximum(target_prediction, axis=1) * (1-done_batch)

        # Set the Q value of the chosen action to the target.
        q_batch = self.model.predict(state_batch)
        q_batch[row_array, action_batch.astype(int)] = target_batch

        # Train the network on the training batch.
        sample_errors = np.abs(q_batch - self.critic_network(state_batch))
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
                    self.critic_target1.set_weights(self.model.get_weights())
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
