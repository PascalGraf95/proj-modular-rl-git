#!/usr/bin/env python

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from ..misc.network_constructor import construct_network
from tensorflow.keras.models import clone_model
from .agent_blueprint import Agent
from datetime import datetime
import os


class DQNAgent(Agent):
    # Static, algorithm specific Parameters
    TrainingParameterSpace = Agent.TrainingParameterSpace.copy()
    DQNParameterSpace = {
        'ReplayMinSize': int,
        'ReplayCapacity': int,
        'NSteps': int,
        'LearningRate': float,
        'SyncMode': str,
        'SyncSteps': int,
        'Tau': float,
        'DoubleLearning': bool
    }
    TrainingParameterSpace = {**TrainingParameterSpace, **DQNParameterSpace}
    NetworkParameterSpace = [{
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
        'DuelingNetworks': bool,
        'NoisyNetworks': bool
    }]
    ActionType = ['DISCRETE']
    ReplayBuffer = 'memory'
    LearningBehavior = 'OffPolicy'
    NetworkTypes = ['QNetwork']
    Metrics = ['ValueLoss']

    def __init__(self, mode,
                 learning_parameters=None,
                 environment_configuration=None,
                 network_parameters=None,
                 model_path=None):
        # Learning Parameters
        self.sync_mode = learning_parameters.get('SyncMode')
        self.sync_steps = learning_parameters.get('SyncSteps')
        self.tau = learning_parameters.get('Tau')
        self.double_learning = learning_parameters.get('DoubleLearning')
        self.gamma = learning_parameters.get('Gamma')
        self.n_steps = learning_parameters.get('NSteps')
        self.training_step = 0

        # Environment Configuration
        self.observation_shapes = environment_configuration.get('ObservationShapes')
        self.action_shape = environment_configuration.get('ActionShape')

        if mode == 'training':
            # Network Construction
            self.model, self.target_model = self.build_network(network_parameters, environment_configuration)
            # Load Pretrained Models
            if model_path:
                self.load_checkpoint(model_path)
            # Compile Networks
            self.model.compile(loss='mse', optimizer=Adam(learning_parameters.get('LearningRate')))

        elif mode == 'testing':
            assert model_path, "No model path entered."
            self.load_checkpoint(model_path)

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            self.model = load_model(path)
            self.target_model = clone_model(self.model)
        elif os.path.isdir(path):
            file_names = [f for f in os.listdir(path) if f.endswith(".h5")]
            for file_name in file_names:
                if "DQN" in file_name:
                    self.model = load_model(os.path.join(path, file_name))
                    self.target_model = clone_model(self.model)
            if not self.model:
                raise FileNotFoundError("Could not find all necessary model files.")
        else:
            raise NotADirectoryError("Could not find directory or file for loading models.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        model_path = os.path.join(path, "DQN_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward))
        print(model_path)
        self.model.save(model_path)

    @staticmethod
    def get_config():
        config_dict = DQNAgent.__dict__
        return Agent.get_config(config_dict)

    def build_network(self, network_parameters, environment_parameters):
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None]
        network_parameters[0]['TargetNetwork'] = True
        network_parameters[0]['NetworkType'] = self.NetworkTypes[0]
        return construct_network(network_parameters[0])

    def act(self, states):
        agent_num = np.shape(states[0])[0]
        if not agent_num:
            return Agent.get_dummy_action(agent_num, self.action_shape, self.ActionType)
        action_values = self.model.predict(states)
        action = np.expand_dims(np.argmax(action_values, axis=1), axis=1)
        return action

    def learn(self, replay_batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        row_array = np.arange(len(replay_batch))
        not_done_batch = ~done_batch.astype(bool)
        # If the state is not terminal:
        # t = 𝑟 + 𝛾 * 𝑚𝑎𝑥_𝑎′ 𝑄̂(𝑠′,𝑎′) else t = r
        target_prediction = self.target_model.predict(next_state_batch)
        target_batch = reward_batch
        if self.double_learning:
            model_prediction_argmax = np.argmax(self.model.predict(next_state_batch), axis=1)
            target_batch += (self.gamma**self.n_steps) * target_prediction[row_array, model_prediction_argmax] * not_done_batch.astype(int)
        else:
            target_batch += (self.gamma**self.n_steps) * np.amax(target_prediction, axis=1) * not_done_batch.astype(int)

        # Set the Q value of the chosen action to the target.
        q_batch = self.model.predict(state_batch)
        q_batch[row_array, action_batch.astype(int)] = target_batch

        # Train the network on the training batch.
        value_loss = self.model.train_on_batch(state_batch, q_batch)

        # Update target network weights
        self.training_step += 1
        self.sync_models()
        return {'Losses/Loss': value_loss}, 0, self.training_step

    def sync_models(self):
        if self.sync_mode == "hard_sync":
            if not self.training_step % self.sync_steps and self.training_step > 0:
                self.target_model.set_weights(self.model.get_weights())
        elif self.sync_mode == "soft_sync":
            self.target_model.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                           for weights, target_weights in zip(self.model.get_weights(),
                                                                              self.target_model.get_weights())])


if __name__ == '__main__':
    # Test methods for debugging
    print(DQNAgent.get_config())
