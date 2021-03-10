#!/usr/bin/env python

import random
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Concatenate, Input, BatchNormalization, Dropout, Add, Subtract, Lambda
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras.backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Model
from tensorflow import keras
from enum import Enum
from .agent_blueprint import Agent
from tensorflow.keras.models import load_model
from ..misc.network_constructor import construct_network
import tensorflow as tf
import os


class A2CAgent(Agent):
    # Static, algorithm specific Parameters
    TrainingParameterSpace = Agent.TrainingParameterSpace.copy()
    A2CParameterSpace = {
        'LearningRate': float,
        'EntropyBeta': float,
        'ClipGrad': float,
        'TrajectoryNum': int
    }
    # TODO: Enable separation of actor and critic network architecture.
    TrainingParameterSpace = {**TrainingParameterSpace, **A2CParameterSpace}
    NetworkParameterSpace = [{
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
    }]
    ActionType = ['CONTINUOUS', 'DISCRETE']
    ReplayBuffer = 'trajectory'
    LearningBehavior = 'OnPolicy'
    NetworkTypes = ['ActorCritic']
    Metrics = ['PolicyLoss', 'ValueLos']

    def __init__(self, mode,
                 learning_parameters=None,
                 environment_configuration=None,
                 network_parameters=None,
                 model_path=None):
        # Learning Parameters
        self.gamma = learning_parameters.get('Gamma')
        self.clip_grad = learning_parameters.get('ClipGrad')
        self.entropy_beta = learning_parameters.get('EntropyBeta')
        self.batch_size = learning_parameters.get('BatchSize')
        self.training_step = 0

        # Environment Configuration
        self.observation_shapes = environment_configuration.get('ObservationShapes')
        self.action_shape = environment_configuration.get('ActionShape')
        self.action_type = environment_configuration.get('ActionType')

        if mode == 'training':
            # Network Construction
            self.actor_critic = self.build_network(network_parameters, environment_configuration)
            # Load Pretrained Models
            if model_path:
                self.load_checkpoint(model_path)
            # Compile Networks
            self.optimizer = Adam(learning_parameters.get('LearningRate'), clipvalue=self.clip_grad, epsilon=1e-3)

        elif mode == 'testing':
            assert model_path, "No model path entered."
            self.actor_critic = load_model(model_path)

    def build_network(self, network_parameters, environment_parameters):
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        if self.action_type == "DISCRETE":
            network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'), 1]
            network_parameters[0]['OutputActivation'] = ["softmax", None]
        else:
            network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                               environment_parameters.get('ActionShape'), 1]
            network_parameters[0]['OutputActivation'] = ["tanh", "softplus", None]
        network_parameters[0]['NetworkType'] = self.NetworkTypes[0]
        return construct_network(network_parameters[0])

    def act(self, states, mode="training"):
        agent_num = np.shape(states[0])[0]
        if not agent_num:
            return Agent.get_dummy_action(agent_num, self.action_shape, self.action_type)
        if self.action_type == "DISCRETE":
            action_probs = self.actor_critic.predict(states)[0]
            actions = np.zeros((agent_num, 1))
            for idx in range(agent_num):
                actions[idx] = np.random.choice(self.action_shape[0], p=action_probs[idx])
        else:
            action_mean, action_var = self.actor_critic.predict(states)[:2]
            if mode == "training":
                std = np.sqrt(action_var)
                actions = np.random.normal(action_mean, std)
            else:
                actions = action_mean
            actions = np.clip(actions, -1, 1)
        return actions

    def custom_loss(self, states, action_batch, discounted_return):
        # Discrete Actions
        if self.action_type == "DISCRETE":
            action_probs, values = self.actor_critic(states)

            # Actor Loss
            log_action_probs = K.log(K.sum(action_probs * action_batch, axis=1))
            advantage = discounted_return - values
            policy_loss = - log_action_probs * advantage
            policy_loss = K.mean(policy_loss)

            # Entropy Loss
            entropy = -self.entropy_beta*K.mean(-K.sum(action_probs * K.log(action_probs), axis=1))

        # Continuous Actions
        else:
            action_mean, action_var, values = self.actor_critic(states)

            # Actor Loss Calculation
            pi_log = - K.square(action_batch - action_mean)
            pi_log /= 2 * K.clip(action_var, 1e-3, 100)
            pi_log -= K.log(K.sqrt(2 * np.pi * action_var))

            advantage = discounted_return - values
            policy_loss = pi_log * advantage
            policy_loss = -K.mean(policy_loss)

            # Entropy Loss
            entropy = self.entropy_beta * K.mean(-(K.log(2*np.pi*action_var)+1)/2)

        # Critic Loss Calculation
        value_loss = K.mean(K.square(values - discounted_return))

        # Combine Loss
        loss = policy_loss + value_loss + entropy

        return loss, policy_loss, value_loss, entropy

    def learn(self, replay_batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        return_batch = np.zeros((len(replay_batch), 1))

        # Loop through the memory batch
        for idx, transition in enumerate(replay_batch):
            return_batch[idx] = transition['reward']
            #if self.action_type == "DISCRETE":
            #    action = np.array(keras.utils.to_categorical(action, self.action_shape))
            #else:
            #    action = np.array(action)
            # action_batch[idx] = action

        with tf.GradientTape() as tape:
            loss, policy_loss, value_loss, entropy = self.custom_loss(state_batch, action_batch, return_batch)
        grads = tape.gradient(loss, self.actor_critic.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.actor_critic.trainable_variables))
        self.training_step += 1
        return {'Losses/Loss': loss, 'Losses/PolicyLoss': policy_loss,
                'Losses/ValueLoss': value_loss, 'Losses/Entropy': entropy}, 0, self.training_step

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            self.actor_critic = load_model(path)
        elif os.path.isdir(path):
            file_names = [f for f in os.listdir(path) if f.endswith(".h5")]
            for file_name in file_names:
                if "A2C" in file_name:
                    self.actor_critic = load_model(os.path.join(path, file_name))
            if not self.actor_critic:
                raise FileNotFoundError("Could not find all necessary model files.")
        else:
            raise NotADirectoryError("Could not find directory or file for loading models.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        self.actor_critic.save(
            os.path.join(path, "A2C_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))

    @staticmethod
    def get_config():
        config_dict = A2CAgent.__dict__
        return Agent.get_config(config_dict)


