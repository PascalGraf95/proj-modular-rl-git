#!/usr/bin/env python

import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras.backend as K
from .agent_blueprint import Agent
from tensorflow.keras.models import load_model
from ..misc.network_constructor import construct_network
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.models import clone_model
import tensorflow_probability as tfp
import os
import math
tfd = tfp.distributions


class SACAgent(Agent):
    # Static, algorithm specific Parameters
    TrainingParameterSpace = Agent.TrainingParameterSpace.copy()
    SACParameterSpace = {
        'LearningRateActor': float,
        'LearningRateCritic': float,
        'ActorTrainingFreq': int,
        'ReplayMinSize': int,
        'ReplayCapacity': int,
        'NSteps': int,
        'SyncMode': str,
        'SyncSteps': int,
        'Tau': float,
        'ClipGrad': float,
        'PrioritizedReplay': bool,
        'LogAlpha': float
    }
    TrainingParameterSpace = {**TrainingParameterSpace, **SACParameterSpace}
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
    }, {
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
    }]
    ActionType = ['CONTINUOUS']
    ReplayBuffer = 'memory'
    LearningBehavior = 'OffPolicy'
    NetworkTypes = ['Actor', 'Critic1', 'Critic2']
    Metrics = ['PolicyLoss', 'ValueLoss', 'AlphaLoss']

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
        self.actor_training_freq = learning_parameters.get('ActorTrainingFreq')
        self.tau = learning_parameters.get('Tau')
        self.epsilon = 1.0e-6
        self.n_steps = learning_parameters.get('NSteps')
        self.clip_grad = learning_parameters.get('ClipGrad')
        self.actor_training_counter = 0
        self.training_step = 0

        # Environment Configuration
        self.observation_shapes = environment_configuration.get('ObservationShapes')
        self.action_shape = environment_configuration.get('ActionShape')
        self.action_type = environment_configuration.get('ActionType')

        # Temperature Parameter Configuration
        self.log_alpha = tf.Variable(tf.ones(1)*learning_parameters.get('LogAlpha'),
                                     constraint=lambda x: tf.clip_by_value(x, -5, 20), trainable=True)
        self.target_entropy = -tf.reduce_sum(tf.ones(self.action_shape))
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_parameters.get('LearningRateActor'),
                                                        clipvalue=self.clip_grad)
        self.alpha = tf.exp(self.log_alpha).numpy()

        if mode == 'training':
            # Network Construction
            self.actor,\
                self.critic1, self.critic_target1,\
                self.critic2, self.critic_target2 = self.build_network(network_parameters, environment_configuration)
            # Load Pretrained Models
            if model_path:
                self.load_checkpoint(model_path)

            # Compile Networks
            self.critic1.compile(optimizer=Adam(learning_rate=learning_parameters.get('LearningRateCritic'),
                                                clipvalue=self.clip_grad), loss="mse")
            self.critic2.compile(optimizer=Adam(learning_rate=learning_parameters.get('LearningRateCritic'),
                                                clipvalue=self.clip_grad), loss="mse")
            self.actor_optimizer = Adam(learning_rate=learning_parameters.get('LearningRateActor'),
                                        clipvalue=self.clip_grad)

        elif mode == 'testing':
            assert model_path, "No model path entered."
            self.load_checkpoint(model_path)

    def build_network(self, network_parameters, environment_parameters):
        # Actor
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                           environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None, None]
        network_parameters[0]['NetworkType'] = self.NetworkTypes[0]
        network_parameters[0]['KernelInitializer'] = "RandomUniform"

        # Critic1
        network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                          environment_parameters.get('ActionShape')]
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['NetworkType'] = self.NetworkTypes[1]
        network_parameters[1]['TargetNetwork'] = True
        network_parameters[1]['Vec2Img'] = False
        network_parameters[1]['KernelInitializer'] = "RandomUniform"

        # Critic2
        network_parameters[2] = network_parameters[1].copy()
        network_parameters[2]['NetworkType'] = self.NetworkTypes[2]

        # Build
        actor = construct_network(network_parameters[0])
        critic1, critic_target1 = construct_network(network_parameters[1])
        critic2, critic_target2 = construct_network(network_parameters[2])

        return actor, critic1, critic_target1, critic2, critic_target2

    def act(self, states, mode="training"):
        agent_num = np.shape(states[0])[0]
        if not agent_num:
            return Agent.get_dummy_action(agent_num, self.action_shape, self.action_type)

        mean, log_std = self.actor.predict(states)
        if mode == "training":
            normal = tfd.Normal(mean, tf.exp(log_std))
            actions = np.tanh(normal.sample())
        else:
            actions = np.tanh(mean)
        return actions

    def forward(self, states):
        mean, log_std = self.actor(states)
        log_std = tf.clip_by_value(log_std, -20, 20)
        normal = tfd.Normal(mean, tf.exp(log_std))

        z = normal.sample()
        action = tf.tanh(z)

        log_prob = normal.log_prob(z) - tf.math.log(1 - action**2 + self.epsilon)
        log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        return action, log_prob

    def learn(self, replay_batch):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        # CRITIC TRAINING
        next_actions, next_log_prob = self.forward(next_state_batch)
        # Critic Target Predictions
        critic_target_prediction1 = self.critic_target1([*next_state_batch, next_actions])
        critic_target_prediction2 = self.critic_target2([*next_state_batch, next_actions])
        critic_target_prediction = tf.minimum(critic_target_prediction1, critic_target_prediction2)
        critic_target = (critic_target_prediction - self.alpha * next_log_prob)*(1-done_batch)

        # Train Both Critic Networks
        y = reward_batch + (self.gamma**self.n_steps) * critic_target
        sample_errors = np.abs(y - self.critic1([*state_batch, action_batch]))
        value_loss1 = self.critic1.train_on_batch([*state_batch, action_batch], y)
        value_loss2 = self.critic2.train_on_batch([*state_batch, action_batch], y)
        value_loss = (value_loss1 + value_loss2)/2

        # ACTOR TRAINING
        self.actor_training_counter += 1
        with tf.GradientTape() as tape:
            new_actions, log_prob = self.forward(state_batch)
            critic_prediction1 = self.critic1([*state_batch, new_actions])
            critic_prediction2 = self.critic2([*state_batch, new_actions])
            critic_prediction = tf.minimum(critic_prediction1, critic_prediction2)
            policy_loss = tf.reduce_mean(self.alpha*log_prob - critic_prediction)
        if self.actor_training_counter >= self.actor_training_freq:
            self.actor_training_counter = 0
            actor_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        # TEMPERATURE PARAMETER UPDATE
        with tf.GradientTape() as tape:
            # _, log_prob_a = self.forward(state_batch)
            alpha_loss = tf.reduce_mean(self.log_alpha * (-log_prob - self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        self.alpha = tf.exp(self.log_alpha).numpy()

        if np.isnan(value_loss) or np.isnan(alpha_loss) or np.isnan(policy_loss):
            print("NAN DETECTED")

        self.training_step += 1
        self.sync_models()
        return {'Losses/Loss': policy_loss+value_loss, 'Losses/PolicyLoss': policy_loss,
                'Losses/ValueLoss': value_loss, 'Losses/AlphaLoss': alpha_loss, 'Losses/Alpha': tf.reduce_mean(self.alpha).numpy()}, \
               sample_errors, self.training_step

    def boost_exploration(self):
        self.log_alpha.assign_add(tf.ones(1))
        return True

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            self.actor = load_model(path)
        elif os.path.isdir(path):
            file_names = [f for f in os.listdir(path) if f.endswith(".h5")]
            for file_name in file_names:
                if "Critic1" in file_name:
                    self.critic1 = load_model(os.path.join(path, file_name))
                    self.critic_target1 = clone_model(self.critic1)
                elif "Critic2" in file_name:
                    self.critic2 = load_model(os.path.join(path, file_name))
                    self.critic_target2 = clone_model(self.critic2)
                elif "Actor" in file_name:
                    self.actor = load_model(os.path.join(path, file_name))
            if not self.actor or not self.critic1 or not self.critic2:
                raise FileNotFoundError("Could not find all necessary model files.")
        else:
            raise NotADirectoryError("Could not find directory or file for loading models.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        self.actor.save(
            os.path.join(path, "SAC_Actor_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))
        if save_all_models:
            self.critic1.save(
                os.path.join(path, "SAC_Critic1_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))
            self.critic2.save(
                os.path.join(path, "SAC_Critic2_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))

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

    @staticmethod
    def get_config():
        config_dict = SACAgent.__dict__
        return Agent.get_config(config_dict)


