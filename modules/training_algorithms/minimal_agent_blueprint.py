#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from modules.misc.logger import LocalLogger
from modules.misc.replay_buffer import LocalFIFOBuffer, LocalRecurrentBuffer
from modules.misc.utility import modify_observation_shapes, set_gpu_growth
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from modules.sidechannel.game_results_sidechannel import GameResultsSideChannel
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from modules.misc.model_path_handling import get_model_key_from_dictionary
import ray
import gym
from gym.wrappers import RescaleAction

import os
import csv


class MinimalActor:
    # region Init
    def __init__(self, port: int, interface: str,
                 preprocessing_algorithm: str,
                 preprocessing_path: str,
                 environment_path: str = ""):
        # region --- Instance Parameters ---
        # region - Networks and Network Parameters -
        # Depending on the chosen algorithm an actor utilizes the actor or critic network to determine its next action.
        self.actor_network = None
        self.critic_network = None
        # It also may have a prior version of the actor network stored for self play purposes.
        self.clone_actor_network = None
        # endregion

        # region - Environment -
        # Each actor interacts with a unique copy of the environment, receives states + rewards and sends new actions.
        # The environment configuration contains information such as the state and action shapes. Instead of connecting
        # directly to the environment in Unity, an executable can be exported and connected to via path.
        self.environment = None
        self.environment_configuration = None
        self.environment_path = environment_path
        # endregion

        # region - Local Buffer -
        # The local buffer stores experiences from previously played episodes until these are transferred to the global
        # buffer.
        self.local_buffer = None
        self.minimum_capacity_reached = False
        # endregion

        # region - Recurrent Properties -
        # Acting and learning utilizing a recurrent neural network introduces a new set of parameters and challenges.
        # Among these, is the need to store and reset the hidden states of the LSTM-layers.
        self.recurrent = None
        self.sequence_length = None
        self.lstm_layer = None
        self.lstm_units = None
        self.lstm_state = None
        self.initial_lstm_state = None
        self.clone_lstm_layer = None
        self.clone_lstm_state = None
        self.clone_initial_lstm_state = None
        # endregion

        # region - Local Logger -
        # Just as the local buffer, the local logger stores episode lengths and rewards from previously played episodes
        # until these are transferred to the global logger.
        self.local_logger = None
        # endregion

        # region - Behavior Parameters -
        # Information read from the environment.
        self.behavior_name = None
        self.behavior_clone_name = None
        self.action_shape = None
        self.action_type = None
        self.observation_shapes = None
        self.agent_number = None
        self.agent_id_offset = None
        self.clone_agent_id_offset = None
        # endregion

        # region - Side Channel -
        self.engine_configuration_channel = None
        self.game_result_side_channel = None
        # endregion

        # region - Preprocessing Algorithm -
        self.preprocessing_algorithm = None
        self.preprocessing_path = preprocessing_path
        # endregion

        # region - Prediction Parameters -
        self.gamma = None
        self.n_steps = None
        # endregion

        # region - Misc -
        self.port = port
        # endregion
        # endregion

        # region --- Algorithm Selection ---
        # This section imports the relevant modules corresponding to the chosen interface, exploration algorithm and
        # preprocessing algorithm.
        self.select_agent_interface(interface)
        self.select_preprocessing_algorithm(preprocessing_algorithm)
        # endregion
    # endregion

    # region Environment Connection and Side Channel Communication
    def connect_to_unity_environment(self):
        """
        Connects an actor instance to a Unity instance while considering different types of side channels to
        send/receive additional data. In case of connecting to the Unity Editor directly, this function will
        wait until you press play in Unity or timeout.
        :return: True when connected.
        """
        self.engine_configuration_channel = EngineConfigurationChannel()
        self.game_result_side_channel = GameResultsSideChannel()
        self.environment = UnityEnvironment(file_name=self.environment_path,
                                            side_channels=[self.engine_configuration_channel,
                                                           self.game_result_side_channel],
                                            base_port=self.port)
        self.environment.reset()
        return True

    def connect_to_gym_environment(self):
        """
        Considering the environment path (which is the environment name for OpenAI Gym) connect this actor instance
        to a Gym environment.
        :return: True when connected.
        """
        self.environment = AgentInterface.connect(self.environment_path)
        # Make sure continuous actions are always bound to the same action scaling
        if not (type(self.environment.action_space) == gym.spaces.Discrete):
            self.environment = RescaleAction(self.environment, -1.0, 1.0)
            print("\n\nEnvironment action space rescaled to -1.0...1.0.\n\n")
        AgentInterface.reset(self.environment)
        return True

    def set_unity_parameters(self, **kwargs):
        """
        The Unity engine configuration channel allows for modifying different simulation settings such as the
        simulation speed and the screen resolution given a dictionary with the respective keys.
        """
        self.engine_configuration_channel.set_configuration_parameters(**kwargs)

    def get_side_channel_information(self, side_channel='game_results'):
        """
        Given the side channel name, check if Unity has sent any new information.
        :param side_channel: Name of the side channel
        :return: New side channel information or None
        """
        if side_channel == 'game_results':
            # Get the latest game results from the respective side channel.
            # This returns None if there has no match concluded since the last query. Otherwise, results is a list that
            # consists of two integers, i.e., the scores of both players in the last match.
            return self.game_result_side_channel.get_game_results()
        return None
    # endregion

    # region Property Query
    def is_minimum_capacity_reached(self):
        """
        Returns true when 50 or more samples are stored in the local buffer, i.e. the buffer content is ready for
        being transferred into the global buffer."""
        # ToDo: Determine influence on training speed when changing this parameter.
        return len(self.local_buffer) >= 50

    def get_environment_configuration(self):
        return self.environment_configuration

    def read_environment_configuration(self):
        self.behavior_name, self.behavior_clone_name = AgentInterface.get_behavior_name(self.environment)
        self.action_type = AgentInterface.get_action_type(self.environment)
        self.action_shape = AgentInterface.get_action_shape(self.environment, self.action_type)
        self.observation_shapes = AgentInterface.get_observation_shapes(self.environment)
        self.agent_number, self.agent_id_offset = AgentInterface.get_agent_number(self.environment, self.behavior_name)
        if self.behavior_clone_name:
            _, self.clone_agent_id_offset = AgentInterface.get_agent_number(self.environment, self.behavior_clone_name)
        self.environment_configuration = {"BehaviorName": self.behavior_name,
                                          "BehaviorCloneName": self.behavior_clone_name,
                                          "ActionShape": self.action_shape,
                                          "ActionType": self.action_type,
                                          "ObservationShapes": self.observation_shapes,
                                          "AgentNumber": self.agent_number}

    def read_trainer_configuration(self, trainer_configuration):
        # Read relevant information from the trainer configuration file.
        # Recurrent Parameters
        self.recurrent = trainer_configuration.get("Recurrent")
        self.sequence_length = trainer_configuration.get("SequenceLength")
        # General Learning Parameters
        self.gamma = trainer_configuration.get("Gamma")
        self.n_steps = trainer_configuration.get("NSteps")
    # endregion

    # region Algorithm Selection
    def select_agent_interface(self, interface):
        global AgentInterface
        if interface == "MLAgentsV18":
            from modules.interfaces.mlagents_v18 import MlAgentsV18Interface as AgentInterface

        elif interface == "OpenAIGym":
            from modules.interfaces.openaigym import OpenAIGymInterface as AgentInterface
        else:
            raise ValueError("An interface for {} is not (yet) supported by this trainer. "
                             "You can implement an interface yourself by utilizing the interface blueprint class "
                             "in the respective folder. "
                             "After that add the respective if condition here.".format(interface))

    def select_preprocessing_algorithm(self, preprocessing_algorithm):
        global PreprocessingAlgorithm
        if preprocessing_algorithm == "None":
            from ..preprocessing.preprocessing_blueprint import PreprocessingAlgorithm
        elif preprocessing_algorithm == "SemanticSegmentation":
            from ..preprocessing.semantic_segmentation import SemanticSegmentation as PreprocessingAlgorithm
        elif preprocessing_algorithm == "ArUcoMarkerDetection":
            from ..preprocessing.aruco_marker_detection import ArUcoMarkerDetection as PreprocessingAlgorithm
        else:
            raise ValueError("There is no {} preprocessing algorithm.".format(preprocessing_algorithm))
    # endregion

    # region Module Instantiation
    def instantiate_local_buffer(self, trainer_configuration):
        if self.recurrent:
            self.local_buffer = LocalRecurrentBuffer(capacity=5000,
                                                     agent_num=self.agent_number,
                                                     n_steps=trainer_configuration.get("NSteps"),
                                                     overlap=trainer_configuration.get("Overlap"),
                                                     sequence_length=trainer_configuration.get("SequenceLength"),
                                                     gamma=trainer_configuration.get("Gamma"))
        else:
            self.local_buffer = LocalFIFOBuffer(capacity=5000,
                                                agent_num=self.agent_number,
                                                n_steps=trainer_configuration.get("NSteps"),
                                                gamma=trainer_configuration.get("Gamma"))

    def instantiate_modules(self, trainer_configuration):
        # region --- Trainer & Environment Configuration ---
        self.read_trainer_configuration(trainer_configuration)
        self.read_environment_configuration()
        # endregion

        # region --- Preprocessing Algorithm Instantiation ---
        # Instantiate the Preprocessing algorithm and if necessary update the observation shapes for network
        # construction.
        self.preprocessing_algorithm = PreprocessingAlgorithm(self.preprocessing_path)
        modified_output_shapes = self.preprocessing_algorithm.get_output_shapes(self.environment_configuration)
        self.environment_configuration["ObservationShapes"] = modified_output_shapes
        self.observation_shapes = modified_output_shapes
        # endregion

        # region --- Local Buffer & Logger Instantiation ---
        # Instantiate a local logger and buffer.
        self.local_logger = LocalLogger(agent_num=self.agent_number)
        self.instantiate_local_buffer(trainer_configuration)
        # endregion
        return
    # endregion

    # region Network Construction and Updates
    def load_network(self, path, clone=False):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Recurrent Memory State Update and Reset
    def get_lstm_layers(self):
        """Get the lstm layer in either the actor network if existent, otherwise the critic."""
        # Sometimes the layer names vary, a LSTM layer might be called "lstm" or "lstm_2", etc.
        # Make sure this function does not fail by getting the actual layers name from the network.
        lstm_layer_name = ""
        if self.actor_network:
            for layer in self.actor_network.layers:
                if "lstm" in layer.name:
                    lstm_layer_name = layer.name
            self.lstm_layer = self.actor_network.get_layer(lstm_layer_name)
        elif self.critic_network:
            for layer in self.critic_network.layers:
                if "lstm" in layer.name:
                    lstm_layer_name = layer.name
            self.lstm_layer = self.critic_network.get_layer(lstm_layer_name)

        # Get the number of units as well as states in the respective layer.
        self.lstm_units = self.lstm_layer.units
        self.lstm_state = [np.zeros((self.agent_number, self.lstm_units), dtype=np.float32),
                           np.zeros((self.agent_number, self.lstm_units), dtype=np.float32)]

        # If in self-play there will be a clone of the actor model. Use the same workflow to acquire its
        # lstm layer as well as unit number and state
        if self.behavior_clone_name:
            for layer in self.clone_actor_network.layers:
                if "lstm" in layer.name:
                    lstm_layer_name = layer.name
            self.clone_lstm_layer = self.clone_actor_network.get_layer(lstm_layer_name)
            self.clone_lstm_state = [np.zeros((self.agent_number, self.lstm_units), dtype=np.float32),
                                     np.zeros((self.agent_number, self.lstm_units), dtype=np.float32)]

    def reset_actor_state(self):
        """This function resets the actors LSTM states to random values at the beginning of a new episode."""
        for layer in self.actor_network.layers:
            if "lstm" in layer.name:
                layer.reset_states()
        if self.behavior_clone_name:
            for layer in self.clone_actor_network.layers:
                if "lstm" in layer.name:
                    layer.reset_states()

    def register_terminal_agents(self, terminal_ids, clone=False):
        """
        Reset the hidden and cell state for the lstm layers of agents that are in a terminal episode state
        """
        if not self.recurrent:
            return
        if clone:
            for agent_id in terminal_ids:
                self.clone_lstm_state[0][agent_id] = np.zeros(self.lstm_units, dtype=np.float32)
                self.clone_lstm_state[1][agent_id] = np.zeros(self.lstm_units, dtype=np.float32)
        else:
            for agent_id in terminal_ids:
                self.lstm_state[0][agent_id] = np.zeros(self.lstm_units, dtype=np.float32)
                self.lstm_state[1][agent_id] = np.zeros(self.lstm_units, dtype=np.float32)

    def set_lstm_states(self, agent_ids, clone=False):
        active_agent_number = len(agent_ids)
        if clone:
            clone_lstm_state = [np.zeros((active_agent_number, self.lstm_units), dtype=np.float32),
                                np.zeros((active_agent_number, self.lstm_units), dtype=np.float32)]
            for idx, agent_id in enumerate(agent_ids):
                clone_lstm_state[0][idx] = self.clone_lstm_state[0][agent_id]
                clone_lstm_state[1][idx] = self.clone_lstm_state[1][agent_id]
            self.clone_initial_lstm_state = [tf.convert_to_tensor(clone_lstm_state[0]),
                                             tf.convert_to_tensor(clone_lstm_state[1])]
            self.clone_lstm_layer.get_initial_state = self.get_initial_clone_state
        else:
            lstm_state = [np.zeros((active_agent_number, self.lstm_units), dtype=np.float32),
                          np.zeros((active_agent_number, self.lstm_units), dtype=np.float32)]
            for idx, agent_id in enumerate(agent_ids):
                lstm_state[0][idx] = self.lstm_state[0][agent_id]
                lstm_state[1][idx] = self.lstm_state[1][agent_id]
            self.initial_lstm_state = [tf.convert_to_tensor(lstm_state[0]), tf.convert_to_tensor(lstm_state[1])]
            self.lstm_layer.get_initial_state = self.get_initial_state

    def get_initial_state(self, inputs):
        return self.initial_lstm_state

    def get_initial_clone_state(self, inputs):
        return self.clone_initial_lstm_state

    def update_lstm_states(self, agent_ids, lstm_state, clone=False):
        if not clone:
            for idx, agent_id in enumerate(agent_ids):
                self.lstm_state[0][agent_id] = lstm_state[0][idx]
                self.lstm_state[1][agent_id] = lstm_state[1][idx]
        else:
            for idx, agent_id in enumerate(agent_ids):
                self.clone_lstm_state[0][agent_id] = lstm_state[0][idx]
                self.clone_lstm_state[1][agent_id] = lstm_state[1][idx]
    # endregion

    # region Environment Interaction
    def play_one_step(self):
        # region - Step Acquisition and Pre-Processing -
        # Step acquisition (steps contain states, done_flags and rewards)
        decision_steps, terminal_steps = AgentInterface.get_steps(self.environment, self.behavior_name)
        # Preprocess steps if a respective algorithm has been activated
        decision_steps, terminal_steps = self.preprocessing_algorithm.preprocess_observations(decision_steps,
                                                                                              terminal_steps)
        # Register terminal agents, so the hidden LSTM state is reset
        self.register_terminal_agents([a_id - self.agent_id_offset for a_id in terminal_steps.agent_id])
        # endregion

        # region - Action Determination -
        # OpenAI's Gym environments require explicit call for rendering
        if AgentInterface.get_interface_name() == "OpenAIGym":
            self.environment.render()

        # If no action is returned by the exploration algorithm, act greedily according to the actor network
        actions = self.act(decision_steps.obs,
                           agent_ids=[a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                           mode="testing")
        # endregion

        # region - Clone Behavior -
        # The clone model architecture usually is an exact copy of the actor network. The weights however might be
        # different depending on the provided model path. All steps in terms of step acquisition, preprocessing and
        # state extension have to be done in the same way as for the original actor.
        if self.behavior_clone_name:
            # Step acquisition (steps contain states, done_flags and rewards)
            clone_decision_steps, clone_terminal_steps = AgentInterface.get_steps(self.environment,
                                                                                  self.behavior_clone_name)
            # Preprocess steps if a respective algorithm has been activated
            clone_decision_steps, clone_terminal_steps = self.preprocessing_algorithm.preprocess_observations(
                clone_decision_steps,
                clone_terminal_steps)
            # Register terminal agents, so the hidden LSTM state is reset
            self.register_terminal_agents([a_id - self.clone_agent_id_offset for a_id in clone_terminal_steps.agent_id],
                                          clone=True)
            # Choose the next action either by exploring or exploiting
            clone_actions = self.act(clone_decision_steps.obs,
                                     agent_ids=[a_id - self.clone_agent_id_offset
                                                for a_id in clone_decision_steps.agent_id],
                                     mode="testing", clone=True)
        else:
            clone_actions = None
        # endregion

        # region - Store interactions to local replay buffers and trackers -
        # Add the intrinsic reward to the actual rewards if available.
        reward_terminal = terminal_steps.reward
        reward_decision = decision_steps.reward
        # Append steps and actions to the local replay buffer
        self.local_buffer.add_new_steps(terminal_steps.obs, reward_terminal,
                                        [a_id - self.agent_id_offset for a_id in terminal_steps.agent_id],
                                        step_type="terminal")
        self.local_buffer.add_new_steps(decision_steps.obs, reward_decision,
                                        [a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                                        actions=actions, step_type="decision")

        # Track the rewards in a local logger
        self.local_logger.track_episode(terminal_steps.reward,
                                        [a_id - self.agent_id_offset for a_id in terminal_steps.agent_id],
                                        step_type="terminal")

        self.local_logger.track_episode(decision_steps.reward,
                                        [a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                                        step_type="decision")
        # endregion

        # region - Check for reset condition and send actions to environment -
        # If all agents are in a terminal state reset the environment actively
        if self.local_buffer.check_reset_condition():
            AgentInterface.reset(self.environment)
            self.local_buffer.done_agents.clear()
        # Otherwise, take a step in the environment according to the chosen action
        else:
            try:
                AgentInterface.step_action(self.environment, self.action_type,
                                           self.behavior_name, actions, self.behavior_clone_name, clone_actions)
            except RuntimeError:
                print("RUNTIME ERROR")
        # endregion
        return True

    def act(self, states, agent_ids=None, mode="training", clone=False):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Misc
    def get_new_samples(self):
        if self.is_minimum_capacity_reached():
            return self.local_buffer.sample(-1, reset_buffer=True, random_samples=False)
        else:
            return None, None

    def get_new_stats(self):
        return self.local_logger.get_episode_stats()
    # endregion


class Learner:
    # region Init
    ActionType = []
    NetworkTypes = []

    def __init__(self, trainer_configuration, environment_configuration, model_dictionary=None,
                 clone_model_dictionary=None):
        # region --- Instance Parameters ---
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
        self.network_update_frequency = trainer_configuration.get('NetworkUpdateFrequency')
        self.reward_normalization = trainer_configuration.get('RewardNormalization')

        # Recurrent Parameters
        self.recurrent = trainer_configuration.get('Recurrent')
        self.sequence_length = trainer_configuration.get('SequenceLength')
        self.burn_in = trainer_configuration.get('BurnIn')
        self.batch_size = trainer_configuration.get('BatchSize')

        # Exploration Parameters
        self.reward_feedback = trainer_configuration.get("RewardFeedback")
        self.policy_feedback = trainer_configuration.get("PolicyFeedback")
        self.exploration_degree = trainer_configuration["ExplorationParameters"].get("ExplorationDegree")
        self.intrinsic_exploration = trainer_configuration.get("IntrinsicExploration")

        # Misc
        self.training_step = 0
        self.steps_since_actor_update = 0
        set_gpu_growth()  # Important step to avoid tensorflow OOM errors when running multiprocessing!

        # - Model Weights -
        # Structures to store information about the given pretrained models (and clone models in case of self-play)
        self.model_dictionary = model_dictionary
        self.clone_model_dictionary = clone_model_dictionary
        # endregion
    # endregion

    # region Network Construction and Transfer
    def get_actor_network_weights(self, update_requested):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def get_clone_network_weights(self, update_requested, clone_from_actor=False):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def build_network(self, network_parameters, environment_parameters):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def sync_models(self):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Checkpoints
    def load_checkpoint_from_path_list(self, model_paths, clone=False):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def load_checkpoint_by_mode(self, mode='latest', mode_two=None):
        # Try to load pretrained models if provided. Otherwise, this method does nothing.
        model_key = get_model_key_from_dictionary(self.model_dictionary, mode=mode)
        if model_key:
            self.load_checkpoint_from_path_list(self.model_dictionary[model_key]['ModelPaths'], clone=False)
        # In case of self-play try to load a clone model if provided. If there is a clone model but no distinct
        # path is provided, the agent actor's weights will be utilized.
        if mode_two:
            clone_model_key = get_model_key_from_dictionary(self.clone_model_dictionary, mode=mode_two)
        else:
            clone_model_key = get_model_key_from_dictionary(self.clone_model_dictionary, mode=mode)
        if clone_model_key:
            self.load_checkpoint_from_path_list(self.clone_model_dictionary[clone_model_key]['ModelPaths'],
                                                clone=True)

    def save_checkpoint(self, path, running_average_reward, training_step, checkpoint_condition=True):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Learning
    def learn(self, replay_batch):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    @tf.autograph.experimental.do_not_convert
    def burn_in_mse_loss(self, y_true, y_pred):
        if self.recurrent:
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)[:, self.burn_in:]
        else:
            return tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
    # endregion

    # region Misc
    def update_sequence_length(self, trainer_configuration):
        self.sequence_length = trainer_configuration.get("SequenceLength")
        self.burn_in = trainer_configuration.get("BurnIn")

    def boost_exploration(self):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    @staticmethod
    def get_config(config_dict):
        config_dict = {key: val for (key, val) in config_dict.items()
                       if not key.startswith('__')
                       and not callable(val)
                       and not type(val) is staticmethod
                       }
        return config_dict

    @staticmethod
    def get_dummy_action(agent_num, action_shape, action_type):
        if action_type == "CONTINUOUS":
            return np.random.random((agent_num, action_shape))
        else:
            return np.random.randint(0, action_shape, (agent_num, 1))

    @staticmethod
    def get_training_batch_from_replay_batch(replay_batch, observation_shapes, action_shape):
        state_batch = []
        next_state_batch = []
        for obs_shape in observation_shapes:
            state_batch.append(np.zeros((len(replay_batch), *obs_shape)))
            next_state_batch.append(np.zeros((len(replay_batch), *obs_shape)))
        try:
            action_batch = np.zeros((len(replay_batch), *action_shape))
        except TypeError:
            action_batch = np.zeros((len(replay_batch), action_shape))
        reward_batch = np.zeros((len(replay_batch), 1))
        done_batch = np.zeros((len(replay_batch), 1))

        for idx, transition in enumerate(replay_batch):
            if type(transition) == int:
                print("Warning: Transition with wrong data type. This training step will be skipped.")
                return None, None, None, None, None
            for idx2, (state, next_state) in enumerate(zip(transition['state'], transition['next_state'])):
                state_batch[idx2][idx] = state
                next_state_batch[idx2][idx] = next_state
            action_batch[idx] = transition['action']
            reward_batch[idx] = transition['reward']
            done_batch[idx] = transition['done']
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    @staticmethod
    def get_training_batch_from_recurrent_replay_batch(replay_batch, observation_shapes, action_shape, sequence_length):
        # State and Next State Batches are lists of numpy-arrays
        state_batch = []
        next_state_batch = []
        # Append an array with the correct shape for each part of the observation
        for obs_shape in observation_shapes:
            state_batch.append(np.zeros((len(replay_batch), sequence_length, *obs_shape)))
            next_state_batch.append(np.zeros((len(replay_batch), sequence_length, *obs_shape)))
        try:
            action_batch = np.zeros((len(replay_batch), sequence_length, *action_shape))
        except TypeError:
            action_batch = np.zeros((len(replay_batch), sequence_length, action_shape))

        reward_batch = np.zeros((len(replay_batch), sequence_length, 1))
        done_batch = np.zeros((len(replay_batch), sequence_length, 1))

        # Loop through all sequences in the batch
        for idx_seq, sequence in enumerate(replay_batch):
            if type(sequence) == int:
                print("Warning: Transition with wrong data type. This training step will be skipped .")
                return None, None, None, None, None
            # Loop through all transitions in one sequence
            for idx_trans, transition in enumerate(sequence):
                # Loop through all components per transition
                for idx_comp, (state, next_state) in enumerate(zip(transition['state'], transition['next_state'])):
                    state_batch[idx_comp][idx_seq][idx_trans] = state
                    next_state_batch[idx_comp][idx_seq][idx_trans] = next_state

                action_batch[idx_seq][idx_trans] = transition['action']
                reward_batch[idx_seq][idx_trans] = transition['reward']
                done_batch[idx_seq][idx_trans] = transition['done']
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    @staticmethod
    def value_function_rescaling(x, eps=1e-3):
        return np.sign(x) * (np.sqrt(np.abs(x) + 1) - 1) + eps * x

    @staticmethod
    def inverse_value_function_rescaling(h, eps=1e-3):
        return np.sign(h) * (((np.sqrt(1 + 4 * eps * (np.abs(h) + 1 + eps)) - 1) / (2 * eps)) - 1)
    # endregion

    # region Parameter Validation
    @staticmethod
    def validate_action_space(agent_configuration, environment_configuration):
        # Check for compatibility of environment and agent action space
        if environment_configuration.get("ActionType") not in agent_configuration.get("ActionType"):
            print("The action spaces of the environment and the agent are not compatible.")
            return False
        return True
    # endregion
