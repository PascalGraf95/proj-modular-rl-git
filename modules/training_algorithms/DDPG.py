#!/usr/bin/env python

import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras.backend as K
from .agent_blueprint import Agent
from tensorflow.keras.models import load_model
from ..misc.network_constructor import construct_network
import tensorflow as tf
from tensorflow.keras.models import clone_model
import os

# WARNING: THIS ALGORITHM IMPLEMENTATION IS DEPRECATED AND CURRENTLY NOT FUNCTIONAL

class DDPGAgent(Agent):
    # Static, algorithm specific Parameters
    TrainingParameterSpace = Agent.TrainingParameterSpace.copy()
    DDPGParameterSpace = {
        'LearningRateActor': float,
        'LearningRateCritic': float,
        'ReplayMinSize': int,
        'ReplayCapacity': int,
        'NSteps': int,
        'SyncMode': str,
        'SyncSteps': int,
        'Tau': float,
        'Epsilon': float
    }
    TrainingParameterSpace = {**TrainingParameterSpace, **DDPGParameterSpace}
    NetworkParameterSpace = [{
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
    }, {
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
    }]
    ActionType = ['CONTINUOUS']
    ReplayBuffer = 'memory'
    LearningBehavior = 'OffPolicy'
    NetworkTypes = ['Actor', 'Critic']
    Metrics = ['PolicyLoss', 'ValueLoss']

    def __init__(self, mode,
                 learning_parameters=None,
                 environment_configuration=None,
                 network_parameters=None,
                 model_path=None):
        # Learning Parameters
        self.gamma = learning_parameters.get('Gamma')
        self.batch_size = learning_parameters.get('BatchSize')
        self.sync_mode = learning_parameters.get('SyncMode')
        self.sync_steps = learning_parameters.get('SyncSteps')
        self.tau = learning_parameters.get('Tau')
        self.epsilon = learning_parameters.get('Epsilon')
        self.n_steps = learning_parameters.get('NSteps')
        self.training_step = 0

        # Environment Configuration
        self.observation_shapes = environment_configuration.get('ObservationShapes')
        self.action_shape = environment_configuration.get('ActionShape')
        self.action_type = environment_configuration.get('ActionType')
        self.learning_rate_actor = learning_parameters.get('LearningRateActor')

        if mode == 'training':
            # Network Construction
            self.actor, self.actor_target,\
                self.critic, self.critic_target = self.build_network(network_parameters, environment_configuration)
            # Load Pretrained Models
            if model_path:
                self.load_checkpoint(model_path)
            # Compile Networks
            self.critic.compile(optimizer=Adam(learning_rate=learning_parameters.get('LearningRateCritic')), loss="mse")
            self.actor_optimizer = Adam(learning_rate=learning_parameters.get('LearningRateActor'))

        elif mode == 'testing':
            assert model_path, "No model path entered."
            self.epsilon = 0.0
            self.load_checkpoint(model_path)

    def build_network(self, network_parameters, environment_parameters):
        # Actor
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = ["tanh"]
        network_parameters[0]['NetworkType'] = self.NetworkTypes[0]
        network_parameters[0]['TargetNetwork'] = True
        # Critic
        network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                          environment_parameters.get('ActionShape')]
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['NetworkType'] = self.NetworkTypes[1]
        network_parameters[1]['TargetNetwork'] = True
        network_parameters[1]['Vec2Img'] = True

        # Build
        actor, actor_target = construct_network(network_parameters[0])
        critic, critic_target = construct_network(network_parameters[1])

        return actor, actor_target, critic, critic_target

    def act(self, states, mode="training"):
        agent_num = np.shape(states[0])[0]
        if not agent_num:
            return Agent.get_dummy_action(agent_num, self.action_shape, self.action_type)

        actions = self.actor.predict(states)
        if mode == "training":
            actions += self.epsilon * np.random.normal(size=actions.shape)
        actions = np.clip(actions, -1, 1)
        return actions

    def learn(self, replay_batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        actor_target_prediction = self.actor_target.predict(next_state_batch)
        critic_target_prediction = self.critic_target.predict([next_state_batch, actor_target_prediction])
        critic_target_prediction[done_batch.astype(bool)] = 0.0
        y = reward_batch + (self.gamma**self.n_steps) * critic_target_prediction
        value_loss = self.critic.train_on_batch([state_batch, action_batch], y)

        # ACTOR TRAINING
        with tf.GradientTape() as tape:
            actions = self.actor(state_batch)
            critic_value = self.critic([state_batch, actions])
            policy_loss = -tf.math.reduce_mean(critic_value)
        actor_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.training_step += 1
        self.sync_models()
        return {'Losses/Loss': policy_loss+value_loss, 'Losses/PolicyLoss': policy_loss,
                'Losses/ValueLoss': value_loss}, 0, self.training_step

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            self.actor = load_model(path)
        elif os.path.isdir(path):
            file_names = [f for f in os.listdir(path) if f.endswith(".h5")]
            for file_name in file_names:
                if "Critic" in file_name:
                    self.critic = load_model(os.path.join(path, file_name))
                    self.critic_target = clone_model(self.critic)
                elif "Actor" in file_name:
                    self.actor = load_model(os.path.join(path, file_name))
            if not self.actor or not self.critic:
                raise FileNotFoundError("Could not find all necessary model files.")
        else:
            raise NotADirectoryError("Could not find directory or file for loading models.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        self.actor.save(
            os.path.join(path, "DDPG_Actor_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))
        if save_all_models:
            self.critic.save(
                os.path.join(path, "DDPG_Critic_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))

    def sync_models(self):
        if self.sync_mode == "hard_sync":
            if not self.training_step % self.sync_steps and self.training_step > 0:
                self.actor_target.set_weights(self.actor.get_weights())
                self.critic_target.set_weights(self.critic.get_weights())
        elif self.sync_mode == "soft_sync":
            self.actor_target.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                           for weights, target_weights in zip(self.actor.get_weights(),
                                                                              self.actor_target.get_weights())])
            self.critic_target.set_weights([self.tau * weights + (1.0 - self.tau) * target_weights
                                            for weights, target_weights in zip(self.critic.get_weights(),
                                                                               self.critic_target.get_weights())])
        else:
            raise ValueError("Sync mode unknown.")

    @staticmethod
    def get_config():
        config_dict = DDPGAgent.__dict__
        return Agent.get_config(config_dict)


