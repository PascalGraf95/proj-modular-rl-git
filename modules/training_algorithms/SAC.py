#!/usr/bin/env python

import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow.keras.backend as K
from .agent_blueprint import Actor, Learner
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
import ray
import time
import math
tfd = tfp.distributions
global AgentInterface



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
            return Learner.get_dummy_action(active_agent_number, self.action_shape, self.action_type)
        with tf.device(self.device):
            # In case of a recurrent network, the state input needs an additional time dimension
            if self.recurrent:
                states = [tf.expand_dims(state, axis=1) for state in states]
            mean, log_std = self.actor_network(states)
            if mode == "training":
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
        with tf.device(self.device):
            if self.recurrent:
                mean, log_std = self.actor_prediction_network(next_state_batch)
            else:
                mean, log_std = self.actor_network(next_state_batch)
            normal = tfd.Normal(mean, tf.exp(log_std))
            next_actions = tf.tanh(normal.sample())
            # Critic Target Predictions
            critic_prediction = self.critic_network([*next_state_batch, next_actions])
            critic_target = critic_prediction*(1-done_batch)

            # Train Both Critic Networks
            y = reward_batch + (self.gamma**self.n_steps) * critic_target
            sample_errors = np.abs(y - self.critic_network([*state_batch, action_batch]))
            # In case of a recurrent agent the priority has to be averaged over each sequence according to the
            # formula in the paper
            if self.recurrent:
                eta = 0.9
                sample_errors = eta*np.max(sample_errors, axis=1) + (1-eta)*np.mean(sample_errors, axis=1)
        return sample_errors

    def update_actor_network(self, network_weights):
        if not len(network_weights):
            return
        self.actor_network.set_weights(network_weights[0])
        self.critic_network.set_weights(network_weights[1])
        if self.recurrent:
            self.actor_prediction_network.set_weights(network_weights[0])
        self.network_update_requested = False
        self.new_steps_taken = 0

    def build_network(self, network_parameters, environment_parameters, idx):
        # Actor
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                           environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None, None]
        network_parameters[0]['NetworkType'] = 'ActorCopy{}'.format(idx)
        network_parameters[0]['KernelInitializer'] = "RandomUniform"
        # Recurrent Parameters
        network_parameters[0]['Recurrent'] = self.recurrent
        network_parameters[0]['ReturnSequences'] = False
        network_parameters[0]['Stateful'] = True
        network_parameters[0]['BatchSize'] = self.agent_number

        # Actor for Error Prediction
        network_parameters[2] = network_parameters[0].copy()
        network_parameters[2]['NetworkType'] = 'ActorErrorPrediction{}'.format(idx)
        network_parameters[2]['ReturnSequences'] = True
        network_parameters[2]['Stateful'] = False
        network_parameters[2]['BatchSize'] = None

        # Critic1
        network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                          environment_parameters.get('ActionShape')]
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['NetworkType'] = 'CriticCopy{}'.format(idx)
        network_parameters[1]['TargetNetwork'] = False
        network_parameters[1]['Vec2Img'] = False
        network_parameters[1]['KernelInitializer'] = "RandomUniform"
        # Recurrent Parameters
        network_parameters[1]['Recurrent'] = self.recurrent
        network_parameters[1]['ReturnSequences'] = True
        network_parameters[1]['Stateful'] = False
        network_parameters[1]['BatchSize'] = None

        # Build
        with tf.device(self.device):
            self.actor_network = construct_network(network_parameters[0])
            self.critic_network = construct_network(network_parameters[1])
            if self.recurrent:
                self.actor_prediction_network = construct_network(network_parameters[2])
        return True


@ray.remote(num_gpus=1)
class SACLearner(Learner):
    # region ParameterSpace
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
    # endregion

    def __init__(self, mode, trainer_configuration, environment_configuration, network_parameters, model_path=None):
        super().__init__(trainer_configuration, environment_configuration)

        # Networks
        self.actor_network = None
        self.critic1, self.critic_target1 = None, None
        self.critic2, self.critic_target2 = None, None
        self.epsilon = 1.0e-6

        # Optimizer
        self.actor_optimizer = None
        self.alpha_optimizer = None

        # Temperature Parameter
        self.log_alpha = None
        self.target_entropy = None

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

        # Load trained Models
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

    def get_actor_network_weights(self):
        if not self.is_network_update_requested():
            return []
        self.steps_since_actor_update = 0
        return [self.actor_network.get_weights(), self.critic1.get_weights()]

    def build_network(self, network_parameters, environment_parameters):
        # Actor
        network_parameters[0]['Input'] = environment_parameters.get('ObservationShapes')
        network_parameters[0]['Output'] = [environment_parameters.get('ActionShape'),
                                           environment_parameters.get('ActionShape')]
        network_parameters[0]['OutputActivation'] = [None, None]
        network_parameters[0]['NetworkType'] = self.NetworkTypes[0]
        network_parameters[0]['KernelInitializer'] = "RandomUniform"
        # Recurrent Parameters
        network_parameters[0]['Recurrent'] = self.recurrent
        network_parameters[0]['ReturnSequences'] = True
        network_parameters[0]['Stateful'] = False
        network_parameters[0]['BatchSize'] = None

        # Critic1
        network_parameters[1]['Input'] = [*environment_parameters.get('ObservationShapes'),
                                          environment_parameters.get('ActionShape')]
        network_parameters[1]['Output'] = [1]
        network_parameters[1]['OutputActivation'] = [None]
        network_parameters[1]['NetworkType'] = self.NetworkTypes[1]
        network_parameters[1]['TargetNetwork'] = True
        network_parameters[1]['Vec2Img'] = False
        network_parameters[1]['KernelInitializer'] = "RandomUniform"
        # Recurrent Parameters
        network_parameters[1]['Recurrent'] = self.recurrent
        network_parameters[1]['ReturnSequences'] = True
        network_parameters[1]['Stateful'] = False
        network_parameters[1]['BatchSize'] = None

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
        if self.recurrent:
            log_prob = tf.reduce_sum(log_prob, axis=2, keepdims=True)
        else:
            log_prob = tf.reduce_sum(log_prob, axis=1, keepdims=True)
        return action, log_prob

    @ray.method(num_returns=3)
    def learn(self, replay_batch):
        if not replay_batch:
            return None, None, self.training_step

        if self.recurrent:
            state_batch, action_batch, reward_batch, next_state_batch, done_batch \
                = Learner.get_training_batch_from_recurrent_replay_batch(replay_batch, self.observation_shapes,
                                                                         self.action_shape, self.sequence_length)
        else:
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
        if self.recurrent:
            eta = 0.9
            sample_errors = eta*np.max(sample_errors, axis=1) + (1-eta)*np.mean(sample_errors, axis=1)

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
        self.steps_since_actor_update += 1
        self.sync_models()
        return {'Losses/Loss': policy_loss+value_loss, 'Losses/PolicyLoss': policy_loss,
                'Losses/ValueLoss': value_loss, 'Losses/AlphaLoss': alpha_loss,
                'Losses/Alpha': tf.reduce_mean(self.alpha).numpy()}, sample_errors, self.training_step

    @staticmethod
    def value_function_rescaling(x, eps=1e-3):
        return np.sign(x)*(np.sqrt(np.abs(x)+1)-1)+eps*x

    @staticmethod
    def inverse_value_function_rescaling(h, eps=1e-3):
        return np.sign(h)*(((np.sqrt(1+4*eps*(np.abs(h)+1+eps))-1)/(2*eps))-1)

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

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False,
                        checkpoint_condition=True):
        if not checkpoint_condition:
            return
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



