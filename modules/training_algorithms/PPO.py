#!/usr/bin/env python

import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras.backend as K
from .agent_blueprint import Agent
from tensorflow.keras.models import load_model
from ..misc.network_constructor import construct_network, LogStdLayer
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.models import clone_model
import tensorflow_probability as tfp
import os
from tensorflow.keras.layers import Dense
from tensorflow.keras import initializers
import math
tfd = tfp.distributions


class PPOAgent(Agent):
    """ Proximal Policy Optimization Algorithm
    - https://arxiv.org/abs/1707.06347

    Implementation Details:
    - https://costa.sh/blog-the-32-implementation-details-of-ppo.html
    - https://arxiv.org/abs/2005.12729
    """
    # Static, algorithm specific Parameters
    TrainingParameterSpace = Agent.TrainingParameterSpace.copy()
    PPOParameterSpace = {
        'LearningRateActor': float,
        'LearningRateCritic': float,
        'GAELambda': float,
        'EntropyCoefficient': float,
        'PPOClip': float,
        'ClipGrad': float,
        'NEpochs': int,
        'TrajectoryNum': int
    }
    TrainingParameterSpace = {**TrainingParameterSpace, **PPOParameterSpace}
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
    ReplayBuffer = 'trajectory'
    LearningBehavior = 'OnPolicy'
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
        self.ppo_clip = learning_parameters.get('PPOClip')
        self.gae_lambda = learning_parameters.get('GAELambda')
        self.n_epochs = learning_parameters.get('NEpochs')
        self.entropy_coefficient = learning_parameters.get('EntropyCoefficient')
        self.clip_grad = learning_parameters.get('ClipGrad')
        self.training_step = 0

        # Environment Configuration
        self.observation_shapes = environment_configuration.get('ObservationShapes')
        self.action_shape = environment_configuration.get('ActionShape')
        self.action_type = environment_configuration.get('ActionType')

        if mode == 'training':
            # Network Construction
            self.actor, self.critic = self.build_network(network_parameters, environment_configuration)
            # Load Pretrained Models
            if model_path:
                self.load_checkpoint(model_path)

            # Compile Networks
            self.critic.compile(optimizer=Adam(learning_rate=learning_parameters.get('LearningRateCritic'),
                                               clipvalue=self.clip_grad), loss="mse")
            self.actor_optimizer = Adam(learning_rate=learning_parameters.get('LearningRateActor'),
                                        clipvalue=self.clip_grad)

        elif mode == 'testing':
            assert model_path, "No model path entered."
            self.load_checkpoint(model_path)

    def build_network(self, network_parameters, environment_parameters):
        # Actor
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = ["tanh"]
        network_parameters[0]['NetworkType'] = self.NetworkTypes[0]
        network_parameters[0]['LogStdOutput'] = True
        network_parameters[0]['KernelInitializer'] = "Orthogonal"

        # Critic
        network_parameters[1]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['NetworkType'] = self.NetworkTypes[1]
        network_parameters[1]['TargetNetwork'] = False
        network_parameters[1]['Vec2Img'] = False
        network_parameters[1]['KernelInitializer'] = "Orthogonal"

        # Build
        actor = construct_network(network_parameters[0])
        critic = construct_network(network_parameters[1])

        return actor, critic

    def act(self, states, mode="training"):
        agent_num = np.shape(states[0])[0]
        if not agent_num:
            return Agent.get_dummy_action(agent_num, self.action_shape, self.action_type)

        mean, log_std = self.actor(states)
        std = tf.exp(tf.broadcast_to(log_std, shape=mean.shape))
        dist = tfp.distributions.Normal(mean, std)
        if mode == 'training':
            actions = dist.sample()
        else:
            actions = mean
        actions = np.clip(actions, -1, 1)
        return actions

    def forward(self, states):
        mean, log_std = self.actor(states)
        std = tf.exp(tf.broadcast_to(log_std, shape=mean.shape))
        dist = tfp.distributions.Normal(mean, std)
        return std, dist

    def learn(self, replay_batch):
        state_batch, action_batch, reward_batch, _, done_batch \
            = self.get_training_batch_from_replay_batch(replay_batch, self.observation_shapes, self.action_shape)

        value_batch = self.critic(state_batch)
        last_advantage_estimation = 0.0
        advantage_batch = []
        return_batch = []
        for val, next_val, r, d in zip(reversed(value_batch[:-1]),
                                       reversed(value_batch[1:]),
                                       reversed(reward_batch[1:]),
                                       reversed(done_batch[1:])):
            if d:
                delta = r - val
                last_advantage_estimation = delta
            else:
                delta = r + self.gamma * next_val - val
                last_advantage_estimation = delta + self.gamma * self.gae_lambda * last_advantage_estimation
            advantage_batch.append(last_advantage_estimation)
            return_batch.append(last_advantage_estimation + val)
        advantage_batch = np.array(list(reversed(advantage_batch)))
        return_batch = np.array(list(reversed(return_batch)))

        # Get Old Policy
        # std, dist = self.forward(state_batch)
        # old_log_prob = tf.squeeze(tf.reduce_sum(dist.log_prob(action_batch), axis=1, keepdims=True))
        # Get Old Policy
        mean, log_std = self.actor(state_batch)
        std = tf.exp(tf.broadcast_to(log_std, shape=mean.shape))
        old_log_prob = - tf.square(mean - action_batch)
        old_log_prob /= 2 * std**2 + 1.0e-8
        old_log_prob -= tf.math.log(tf.sqrt(2 * np.pi * std**2))
        #old_log_prob = tf.reduce_mean(old_log_prob, axis=-1)

        state_batch = [s[:-1].copy() for s in state_batch]
        action_batch = action_batch[:-1].copy()
        old_log_prob = old_log_prob[:-1]

        # Stabilize Advantage
        #advantage_batch = advantage_batch - np.mean(advantage_batch)
        #advantage_batch /= np.std(advantage_batch) + 1e-8

        for _ in range(self.n_epochs):
            # Critic Training
            value_loss = self.critic.train_on_batch(state_batch, return_batch)
            # Actor Training
            with tf.GradientTape() as tape:
                # std, dist = self.forward(state_batch)
                # new_log_prob = tf.squeeze(tf.reduce_sum(dist.log_prob(action_batch), axis=1, keepdims=True))
                mean, log_std = self.actor(state_batch)
                std = tf.exp(tf.broadcast_to(log_std, shape=mean.shape))
                new_log_prob = - tf.square(mean - action_batch)
                new_log_prob /= 2 * std**2 + 1.0e-8
                new_log_prob -= tf.math.log(tf.sqrt(2 * np.pi * std**2))
                #new_log_prob = tf.reduce_mean(new_log_prob, axis=-1)

                policy_ratio = tf.exp(new_log_prob - old_log_prob)
                objective = advantage_batch * policy_ratio
                clipped_objective = advantage_batch * tf.clip_by_value(policy_ratio,
                                                                       1 - self.ppo_clip, 1 + self.ppo_clip)
                objective_loss = - tf.reduce_mean(tf.minimum(objective, clipped_objective))
                entropy_loss = tf.reduce_mean(- log_std + 0.5 * tf.math.log(2.0 * np.pi * np.e)) * self.entropy_coefficient
                # ppo_entropy_loss = - tf.reduce_mean(self.entropy_coefficient * dist.entropy())
                policy_loss = objective_loss + entropy_loss
            grad_actor = tape.gradient(policy_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grad_actor, self.actor.trainable_variables))

        self.training_step += 1
        return {'Losses/Std': np.mean(std),
                'Losses/Loss': policy_loss + value_loss,
                'Losses/PolicyLoss': policy_loss,
                'Losses/ValueLoss': value_loss}, 0, self.training_step

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            with tf.keras.utils.CustomObjectScope({'LogStdLayer': LogStdLayer}):
                self.actor = load_model(path)
        elif os.path.isdir(path):
            file_names = [f for f in os.listdir(path) if f.endswith(".h5")]
            for file_name in file_names:
                if "Critic" in file_name:
                    self.critic = load_model(os.path.join(path, file_name))
                elif "Actor" in file_name:
                    with tf.keras.utils.CustomObjectScope({'LogStdLayer': LogStdLayer}):
                        self.actor = load_model(os.path.join(path, file_name))
            if not self.actor or not self.critic:
                raise FileNotFoundError("Could not find all necessary model files.")
        else:
            raise NotADirectoryError("Could not find directory or file for loading models.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        self.actor.save(
            os.path.join(path, "PPO_Actor_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))
        if save_all_models:
            self.critic.save(
                os.path.join(path, "PPO_Critic_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))

    @staticmethod
    def get_config():
        config_dict = PPOAgent.__dict__
        return Agent.get_config(config_dict)


