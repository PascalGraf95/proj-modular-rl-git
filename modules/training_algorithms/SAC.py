#!/usr/bin/env python

import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras.backend as K
from .agent_blueprint import Agent, Actor, Learner
from tensorflow.keras.models import load_model
from ..misc.network_constructor import construct_network
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from ..sidechannel.curriculum_sidechannel import CurriculumSideChannelTaskInfo
from ..misc.replay_buffer import FIFOBuffer
from ..misc.logger import LocalLogger
import tensorflow as tf
from tensorflow.keras import losses
from tensorflow.keras.models import clone_model
import tensorflow_probability as tfp
import os
import math
tfd = tfp.distributions
global AgentInterface
import ray


@ray.remote
class SACActor(Actor):
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
            return Agent.get_dummy_action(active_agent_number, self.action_shape, self.action_type)
        with tf.device(self.device):
            mean, log_std = self.actor_network(states)
            if mode == "training":
                normal = tfd.Normal(mean, tf.exp(log_std))
                actions = tf.tanh(normal.sample())
            else:
                actions = tf.tanh(mean)
        return actions.numpy()

    def get_sample_errors(self, samples):
        state_batch, action_batch, reward_batch, next_state_batch, done_batch \
            = Learner.get_training_batch_from_replay_batch(samples, self.observation_shapes, self.action_shape)
        with tf.device(self.device):
            # CRITIC TRAINING
            next_actions = self.act(next_state_batch)
            # Critic Target Predictions
            critic_prediction = self.critic_network([*next_state_batch, next_actions])
            critic_target = critic_prediction*(1-done_batch)

            # Train Both Critic Networks
            y = reward_batch + (self.gamma**self.n_steps) * critic_target
            sample_errors = np.abs(y - self.critic_network([*state_batch, action_batch]))
        return sample_errors

    def update_actor_network(self, network_weights):
        self.actor_network.set_weights(network_weights[0])
        self.critic_network.set_weights(network_weights[1])
        self.network_update_requested = False
        self.new_steps_taken = 0

    def get_exploration_logs(self, idx):
        return self.exploration_algorithm.get_logs(idx)

    def build_network(self, network_parameters, environment_parameters, idx):
        # Actor
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                           environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None, None]
        network_parameters[0]['NetworkType'] = 'ActorCopy{}'.format(idx)
        network_parameters[0]['KernelInitializer'] = "RandomUniform"

        # Critic1
        network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                          environment_parameters.get('ActionShape')]
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['NetworkType'] = 'CriticCopy{}'.format(idx)
        network_parameters[1]['TargetNetwork'] = False
        network_parameters[1]['Vec2Img'] = False
        network_parameters[1]['KernelInitializer'] = "RandomUniform"

        # Build
        with tf.device(self.device):
            self.actor_network = construct_network(network_parameters[0])
            self.critic_network = construct_network(network_parameters[1])
        return True


@ray.remote(num_gpus=1)
class SACLearner(Learner):
    # Static, algorithm specific Parameters
    TrainingParameterSpace = Learner.TrainingParameterSpace.copy()
    SACParameterSpace = {
        'LearningRateActor': float,
        'LearningRateCritic': float,
        'LogAlpha': float
    }
    TrainingParameterSpace = {**TrainingParameterSpace, **SACParameterSpace}
    NetworkParameterSpace = [{
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
    },
        {
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
    },
        {
        'VisualNetworkArchitecture': str,
        'VectorNetworkArchitecture': str,
        'Units': int,
        'Filters': int,
    }]
    ActionType = ['CONTINUOUS']
    NetworkTypes = ['Actor', 'Critic1', 'Critic2']
    Metrics = ['PolicyLoss', 'ValueLoss', 'AlphaLoss']

    def __init__(self, mode, trainer_configuration, environment_configuration, network_parameters, model_path=None):
        self.actor_network = None
        self.critic1, self.critic_target1 = None, None
        self.critic2, self.critic_target2 = None, None
        self.epsilon = 1.0e-6

        self.actor_optimizer = None
        self.alpha_optimizer = None

        self.log_alpha = None
        self.target_entropy = None

        self.action_shape = environment_configuration.get('ActionShape')
        self.observation_shapes = environment_configuration.get('ObservationShapes')

        self.n_steps = trainer_configuration.get('NSteps')
        self.gamma = trainer_configuration.get('Gamma')
        self.sync_mode = trainer_configuration.get('SyncMode')
        self.sync_steps = trainer_configuration.get('SyncSteps')
        self.tau = trainer_configuration.get('Tau')
        self.clip_grad = trainer_configuration.get('ClipGrad')

        self.training_step = 0

        self.critic1, self.critic_target1, self.critic2, \
            self.critic_target2, self.actor_network = None, None, None, None, None

        self.set_gpu_growth()
        # Construct or load the required neural networks based on the trainer configuration and environment information
        if mode == 'training':
            # Network Construction
            self.build_network(network_parameters, environment_configuration)
            # Load Pretrained Models
            if model_path:
                self.load_checkpoint(model_path)

            # Compile Networks
            self.critic1.compile(optimizer=Adam(learning_rate=trainer_configuration.get('LearningRateCritic'),
                                                clipvalue=self.clip_grad), loss="mse")
            self.critic2.compile(optimizer=Adam(learning_rate=trainer_configuration.get('LearningRateCritic'),
                                                clipvalue=self.clip_grad), loss="mse")
            self.actor_optimizer = Adam(learning_rate=trainer_configuration.get('LearningRateActor'),
                                        clipvalue=self.clip_grad)

        elif mode == 'testing':
            assert model_path, "No model path entered."
            self.load_checkpoint(model_path)

        # Temperature Parameter Configuration
        self.log_alpha = tf.Variable(tf.ones(1)*trainer_configuration.get('LogAlpha'),
                                     constraint=lambda x: tf.clip_by_value(x, -10, 20), trainable=True)
        self.target_entropy = -tf.reduce_sum(tf.ones(self.action_shape))
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=trainer_configuration.get('LearningRateActor'),
                                                        clipvalue=self.clip_grad)
        self.alpha = tf.exp(self.log_alpha).numpy()

    def set_gpu_growth(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # tf.config.experimental.set_virtual_device_configuration(gpu)
                # , [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)]

    def get_actor_network_weights(self):
        return [self.actor_network.get_weights(), self.critic1.get_weights()]

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
        self.actor_network = construct_network(network_parameters[0])
        self.critic1, self.critic_target1 = construct_network(network_parameters[1])
        self.critic2, self.critic_target2 = construct_network(network_parameters[2])

    def forward(self, states):
        mean, log_std = self.actor_network(states)
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
        with tf.GradientTape() as tape:
            new_actions, log_prob = self.forward(state_batch)
            critic_prediction1 = self.critic1([*state_batch, new_actions])
            critic_prediction2 = self.critic2([*state_batch, new_actions])
            critic_prediction = tf.minimum(critic_prediction1, critic_prediction2)
            policy_loss = tf.reduce_mean(self.alpha*log_prob - critic_prediction)

        actor_grads = tape.gradient(policy_loss, self.actor_network.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_network.trainable_variables))

        # TEMPERATURE PARAMETER UPDATE
        with tf.GradientTape() as tape:
            alpha_loss = tf.reduce_mean(self.log_alpha * (-log_prob - self.target_entropy))

        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
        self.alpha = tf.exp(self.log_alpha).numpy()

        if np.isnan(value_loss) or np.isnan(alpha_loss) or np.isnan(policy_loss):
            print("NAN DETECTED")

        self.training_step += 1
        self.sync_models()
        return {'Losses/Loss': policy_loss+value_loss, 'Losses/PolicyLoss': policy_loss,
                'Losses/ValueLoss': value_loss, 'Losses/AlphaLoss': alpha_loss,
                'Losses/Alpha': tf.reduce_mean(self.alpha).numpy()}, sample_errors, self.training_step

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
        if os.path.isfile(path):
            self.actor_network = load_model(path)
        elif os.path.isdir(path):
            file_names = [f for f in os.listdir(path) if f.endswith(".h5")]
            for file_name in file_names:
                if "Critic1" in file_name:
                    self.critic1 = load_model(os.path.join(path, file_name))
                    self.critic_target1 = clone_model(self.critic1)
                    self.critic_target1.set_weights(self.critic1.get_weights())
                elif "Critic2" in file_name:
                    self.critic2 = load_model(os.path.join(path, file_name))
                    self.critic_target2 = clone_model(self.critic2)
                    self.critic_target2.set_weights(self.critic2.get_weights())
                elif "Actor" in file_name:
                    self.actor_network = load_model(os.path.join(path, file_name))
            if not self.actor_network or not self.critic1 or not self.critic2:
                raise FileNotFoundError("Could not find all necessary model files.")
        else:
            raise NotADirectoryError("Could not find directory or file for loading models.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        self.actor_network.save(
            os.path.join(path, "SAC_Actor_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))
        if save_all_models:
            self.critic1.save(
                os.path.join(path, "SAC_Critic1_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))
            self.critic2.save(
                os.path.join(path, "SAC_Critic2_Step{}_Reward{:.2f}.h5".format(training_step, running_average_reward)))

    def boost_exploration(self):
        self.log_alpha = tf.Variable(tf.ones(1)*-0.7,
                                     constraint=lambda x: tf.clip_by_value(x, -10, 20), trainable=True)
        return True

    @staticmethod
    def get_config():
        config_dict = SACLearner.__dict__
        return Learner.get_config(config_dict)



