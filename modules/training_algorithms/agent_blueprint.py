#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from modules.misc.logger import LocalLogger
from modules.misc.replay_buffer import LocalFIFOBuffer, LocalRecurrentBuffer
from modules.misc.utility import modify_observation_shapes, set_gpu_growth
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from modules.sidechannel.game_results_sidechannel import GameResultsSideChannel
from modules.sidechannel.environment_info_sidechannel import EnvironmentInfoSideChannel
from modules.curriculum_strategies.curriculum_strategy_blueprint import CurriculumCommunicator
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from modules.misc.model_path_handling import get_model_key_from_dictionary
import ray
import gym
from gym.wrappers import RescaleAction

import os
import csv


class Actor:
    # region Init
    def __init__(self, idx: int, port: int, mode: str,
                 interface: str,
                 preprocessing_algorithm: str,
                 preprocessing_path: str,
                 exploration_algorithm: str,
                 environment_path: str = "",
                 demonstration_path: str = "",
                 device: str = '/cpu:0'):
        # region --- Instance Parameters ---
        # region - Networks and Network Parameters -
        # Depending on the chosen algorithm an actor utilizes the actor or critic network to determine its next action.
        self.actor_network = None
        self.critic_network = None
        # It also may have a prior version of the actor network stored for self play purposes.
        self.clone_actor_network = None
        # If the actor is working in a recurrent fashion another copy of the actor network exists.
        self.actor_prediction_network = None
        self.critic_prediction_network = None
        # Lastly, there are values that can optionally be fed back to the network as inputs in addition to the
        # observation received by the respective environment. Such inputs are the prior action, the prior rewards and
        # the exploration policy used by the agent during the prior step. However, this optional feedback logic is only
        # compatible with recurrent agents.
        self.action_feedback = None
        self.policy_feedback = None
        self.reward_feedback = None
        # The following parameters keep track of the network's update status, i.e. if a new version of the networks is
        # requested from the learner and how long it's been since the last update
        self.network_update_requested = False
        self.steps_taken_since_network_update = 0
        self.network_update_frequency = 1000
        self.clone_network_update_frequency = 1000
        self.last_clone_update_step = -1
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

        # region - Tensorflow Device -
        # Working in an async fashion using ray makes it necessary to actively distribute processes among the CPU and
        # GPUs available. Each actor is usually placed on an individual CPU core/thread.
        self.device = device
        # endregion

        # region - CQL -

        self.samples_buffered = False
        self.demonstration_path = demonstration_path
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

        # region - Exploration Algorithm -
        self.exploration_configuration = None
        self.exploration_algorithm = None
        self.adaptive_exploration = False
        self.intrinsic_exploration = False
        self.additional_network_inputs = False
        self.prior_intrinsic_reward = 0
        self.prior_extrinsic_reward = 0
        self.prior_action = None
        self.exploration_degree = None
        self.exploration_policy_idx = idx
        self.reward_correction_factor = 1
        self.steps_since_update = 0
        self.episode_begin = True
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
        self.mode = mode
        self.port = port
        self.index = idx
        # endregion
        # endregion

        # region --- Algorithm Selection ---
        # This section imports the relevant modules corresponding to the chosen interface, exploration algorithm and
        # preprocessing algorithm.
        self.select_agent_interface(interface)
        self.select_exploration_algorithm(exploration_algorithm)
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
        self.environment_info_side_channel = EnvironmentInfoSideChannel()
        self.environment = UnityEnvironment(file_name=self.environment_path,
                                            side_channels=[self.engine_configuration_channel,
                                                           self.game_result_side_channel, 
                                                           self.environment_info_side_channel],
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
        elif side_channel == 'environment_info':
            # Get the latest environment information from the respective side channel.
            # This returns None if no environment information are retrieved. Otherwise, it returns
            # a dictionary with the environment info like reward composition.
            return self.environment_info_side_channel.get_environment_information_from_string()
        return None
    # endregion

    # region Self Play Rating
    def update_history_with_latest_game_results(self, history_path, player_keys):
        """
        This function gets the latest game results from the respective Unity side channel, then prompts the update
        of the game history and induces an update of the actor models when appropriate.
        :return: True if a result has been appended to the result history.
        """
        # Read the latest game results from the side channel. If existent, induce an update of the game history.
        # Else, return False.
        game_result = self.get_side_channel_information('game_results')
        if game_result:
            self.append_to_game_results_history(game_result, history_path, player_keys)
            return True
        return False

    @staticmethod
    def append_to_game_results_history(game_result, history_path, player_keys):
        """
        Appends a new game result to an either existing or new csv files along with the keys of the two players in the
        respective fixture.
        :param game_result: List of two integers describing a games outcome in self-play
        :param history_path: Path to a csv file of game results (otherwise will be created)
        :param player_keys: Keys of the players belonging to the game results
        :return:
        """
        # Check if there already exists a respective history file of played games.
        # Otherwise, create it.
        if not os.path.isfile(history_path):
            with open(history_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["game_id", "player_key_a", "player_key_b", "score_a", "score_b"])
        # If so just append the latest game results to that file along with the player keys
        with open(history_path) as file:
            # To get the game id, just count the number of rows.
            reader = csv.DictReader(file)
            game_id = sum(1 for _ in reader)

        with open(history_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([game_id, player_keys[0], player_keys[1], game_result[0], game_result[1]])
    # endregion

    # region Property Query
    def is_minimum_capacity_reached(self):
        """
        Returns true when 50 or more samples are stored in the local buffer, i.e. the buffer content is ready for
        being transferred into the global buffer."""
        # ToDo: Determine influence on training speed when changing this parameter.
        return len(self.local_buffer) >= 50

    def is_network_update_requested(self):
        """
        Checks if enough steps have been taken since the last network update so new weights are requested.
        :return: True when network update is requested, false otherwise
        """
        return self.steps_taken_since_network_update >= self.network_update_frequency

    def is_clone_network_update_requested(self, total_episodes):
        """
        Checks if enough episodes have been taken since the last clone network update so new weights are requested.
        :param total_episodes:
        :return:
        """
        if total_episodes != self.last_clone_update_step:
            if total_episodes == 0:
                self.last_clone_update_step = total_episodes
                return True
            elif self.clone_network_update_frequency:
                if total_episodes % self.clone_network_update_frequency == 0:
                    self.last_clone_update_step = total_episodes
                    return True
        return False

    def get_exploration_logs(self):
        return self.exploration_algorithm.get_logs()

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
        # Exploration Parameters
        self.adaptive_exploration = trainer_configuration.get("AdaptiveExploration")
        self.exploration_degree = trainer_configuration["ExplorationParameters"].get("ExplorationDegree")
        # Additional information that can be fed back to the network as inputs
        self.action_feedback = trainer_configuration.get("ActionFeedback")
        self.reward_feedback = trainer_configuration.get("RewardFeedback")
        self.policy_feedback = trainer_configuration.get("PolicyFeedback")

    def get_exploration_configuration(self):
        """Gather the parameters requested by selected the exploration algorithm.
        :return: None
        """
        self.exploration_configuration = ExplorationAlgorithm.get_config()
        return self.exploration_configuration
    # endregion

    # region Algorithm Selection
    def select_agent_interface(self, interface):
        global AgentInterface
        if interface == "MLAgentsV18":
            from modules.interfaces.mlagents_v20 import MlAgentsV20Interface as AgentInterface

        elif interface == "OpenAIGym":
            from modules.interfaces.openaigym import OpenAIGymInterface as AgentInterface
        else:
            raise ValueError("An interface for {} is not (yet) supported by this trainer. "
                             "You can implement an interface yourself by utilizing the interface blueprint class "
                             "in the respective folder. "
                             "After that add the respective if condition here.".format(interface))

    def select_exploration_algorithm(self, exploration_algorithm):
        global ExplorationAlgorithm
        if exploration_algorithm == "EpsilonGreedy":
            from ..exploration_algorithms.epsilon_greedy import EpsilonGreedy as ExplorationAlgorithm
        elif exploration_algorithm == "None":
            from ..exploration_algorithms.exploration_algorithm_blueprint import ExplorationAlgorithm
        elif exploration_algorithm == "ICM":
            from ..exploration_algorithms.intrinsic_curiosity_module import \
                IntrinsicCuriosityModule as ExplorationAlgorithm
        elif exploration_algorithm == "RND":
            from ..exploration_algorithms.random_network_distillation import RandomNetworkDistillation as ExplorationAlgorithm
            self.intrinsic_exploration = True
        elif exploration_algorithm == "ENM":
            from ..exploration_algorithms.episodic_novelty_module import EpisodicNoveltyModule as ExplorationAlgorithm
            self.intrinsic_exploration = True
        elif exploration_algorithm == "NGU":
            from ..exploration_algorithms.never_give_up import NeverGiveUp as ExplorationAlgorithm
            self.intrinsic_exploration = True
        else:
            raise ValueError("There is no {} exploration algorithm.".format(exploration_algorithm))

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
        self.network_update_frequency = trainer_configuration.get("NetworkUpdateFrequency")
        self.clone_network_update_frequency = trainer_configuration.get("SelfPlayNetworkUpdateFrequency")

    def update_sequence_length(self, trainer_configuration):
        self.sequence_length = trainer_configuration.get("SequenceLength")
        self.local_buffer.reset(self.sequence_length)

    def instantiate_modules(self, trainer_configuration):
        # region --- Trainer & Environment Configuration ---
        self.read_trainer_configuration(trainer_configuration)
        self.read_environment_configuration()
        # endregion

        # region --- Exploration Algorithm Instantiation ---
        # Instantiate the Exploration Algorithm according to the environment and training configuration.
        self.exploration_algorithm = ExplorationAlgorithm(self.environment_configuration["ActionShape"],
                                                          self.environment_configuration["ObservationShapes"],
                                                          self.environment_configuration["ActionType"],
                                                          trainer_configuration["ExplorationParameters"],
                                                          trainer_configuration,
                                                          self.index)
        # endregion

        # region --- Observation Extensions ---
        # Initialize prior actions randomly and prior rewards to 0, if action or reward feedback is enabled.
        self.reset_observation_extensions()

        # Extend observation shapes given by environment through additional network inputs.
        modified_observation_shapes = modify_observation_shapes(self.environment_configuration["ObservationShapes"],
                                                                self.action_shape, self.action_type,
                                                                self.action_feedback, self.reward_feedback,
                                                                self.policy_feedback)
        self.environment_configuration["ObservationShapes"] = modified_observation_shapes
        self.observation_shapes = modified_observation_shapes
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
        self.modify_local_buffer_gamma()
        # endregion
        return
    # endregion

    # region Network Construction and Updates
    def build_network(self, network_parameters, environment_parameters):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def update_actor_network(self, network_weights):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def update_clone_network(self, network_weights):
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
    def play_one_step(self, training_step):
        # region - Step Acquisition and Pre-Processing -
        # Step acquisition (steps contain states, done_flags and rewards)
        decision_steps, terminal_steps = AgentInterface.get_steps(self.environment, self.behavior_name)
        # Preprocess steps if a respective algorithm has been activated
        decision_steps, terminal_steps = self.preprocessing_algorithm.preprocess_observations(decision_steps,
                                                                                              terminal_steps)
        # Register terminal agents, so the hidden LSTM state is reset
        self.register_terminal_agents([a_id - self.agent_id_offset for a_id in terminal_steps.agent_id])
        # endregion

        # region - Exploration and Action Determination -
        # Set/Reset variables and algorithms when agent reaches episodic borders
        self.episodic_border_routine(terminal_steps)
        # There are exploration algorithms that calculate intrinsic rewards each timestep
        intrinsic_reward = self.calculate_intrinsic_reward(decision_steps, terminal_steps)

        if self.mode != "training":
            # OpenAI's Gym environments require explicit call for rendering
            if AgentInterface.get_interface_name() == "OpenAIGym":
                self.environment.render()

            # ToDo: Parse the intrinsic reward as well as the exploration policy index from the saved model to the
            #       model constructor. Until then 0, 0 will be utilized (exploration_policy_idx is set in constructor)
            intrinsic_reward = 0

        # In case these options have been enabled, the states are augmented with additional information, more precisely:
        # the last chosen action, the last extrinsic & intrinsic reward as well as the exploration policy index.
        decision_steps, terminal_steps = self.extend_observations(decision_steps, terminal_steps)
        # Try to obtain a new action from the exploration algorithm.
        actions = self.act_according_to_exploration_algorithm(decision_steps, terminal_steps)
        # If no action is returned by the exploration algorithm, act greedily according to the actor network
        if actions is None:
            actions = self.act(decision_steps.obs,
                               agent_ids=[a_id - self.agent_id_offset for a_id in decision_steps.agent_id],
                               mode=self.mode)
        # Save the current values to append to the state in the next iteration via self.extend_observations
        self.update_prior_metrics(intrinsic_reward, actions, decision_steps, terminal_steps)
        # endregion

        # region - Clone Behavior -
        # The clone model architecture usually is an exact copy of the actor network. The weights however might be
        # different depending on the provided model path. All steps in terms of step acquisition, preprocessing and
        # state extension have to be done in the same way as for the original actor.
        # ToDo: Add Border Routine, Intrinsic Reward Calculation and State Extensions to the clone model region.
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
                                     mode=self.mode, clone=True)
        else:
            clone_actions = None
        # endregion

        # region - Store interactions to local replay buffers and trackers -
        # Add the intrinsic reward to the actual rewards if available.
        reward_terminal = terminal_steps.reward + intrinsic_reward
        reward_decision = decision_steps.reward + intrinsic_reward
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
        # If enough steps have been taken, mark agent ready for updated network
        self.steps_taken_since_network_update += len(decision_steps)
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
    def exploration_learning_step(self, samples):
        if not samples:
            return
        if self.adaptive_exploration:
            self.exploration_algorithm.learning_step(samples)

    def get_intrinsic_rewards(self, samples):
        if not samples:
            return samples
        with tf.device(self.device):
            samples = self.exploration_algorithm.get_intrinsic_reward(samples)
        return samples

    @ray.method(num_returns=2)
    def get_new_samples(self):
        if self.is_minimum_capacity_reached():
            return self.local_buffer.sample(-1, reset_buffer=True, random_samples=False)
        else:
            return None, None

    @ray.method(num_returns=3)
    def get_new_stats(self):
        return self.local_logger.get_episode_stats()

    def get_sample_errors(self, samples):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Additional Exploration Functions
    def episodic_border_routine(self, terminal_steps):
        """
        Reset specific variables and algorithms when agent reaches episodic begin or ending.

        Parameters
        ----------
        terminal_steps:
            Contains the observation values, reward value and further information about the current step if terminal.
        """
        # At the beginning of each episode certain values have to be (re)set. The prior action and rewards which
        # can be utilized to extend the original observations are set to a random value or 0 respectively.
        if self.episode_begin:
            self.reset_observation_extensions()
            self.episode_begin = False

        # Episode ending reached
        if len(terminal_steps.obs[0]):
            self.episode_begin = True
            with tf.device(self.device):
                # Reset exploration algorithm
                self.exploration_algorithm.reset()

    def reset_observation_extensions(self):
        # Reset the prior actions to random values.
        if self.action_feedback:
            if self.action_type == "DISCRETE":
                self.prior_action = np.random.randint(0, self.action_shape, (1,))
            else:
                self.prior_action = np.random.uniform(-1.0, 1.0, (self.action_shape,))
        # Reset the intrinsic as well es extrinsic reward to 0.
        if self.reward_feedback:
            self.prior_intrinsic_reward = 0
            self.prior_extrinsic_reward = 0

    def modify_local_buffer_gamma(self):
        # Set local_buffer's gamma and calculate the individual gamma list per actor
        if self.mode == "training" and self.intrinsic_exploration:
            self.local_buffer.gamma = self.exploration_degree[self.exploration_policy_idx]["gamma"]
            self.local_buffer.gamma_list = [self.local_buffer.gamma ** n for n in range(self.n_steps)]

    def act_according_to_exploration_algorithm(self, decision_steps, terminal_steps):
        if self.mode == "testing" or self.mode == "fastTesting":
            return None
        if self.intrinsic_exploration:
            actions = self.exploration_algorithm.epsilon_greedy(decision_steps)
        else:
            with tf.device(self.device):
                actions = self.exploration_algorithm.act(decision_steps, terminal_steps)
        return actions

    def calculate_intrinsic_reward(self, decision_steps, terminal_steps):
        if self.intrinsic_exploration and self.mode == "training":
            with tf.device(self.device):
                # Calculate intrinsic reward
                intrinsic_reward = self.exploration_algorithm.act(decision_steps, terminal_steps)
                # Scale the intrinsic reward through exploration policies' beta.
                intrinsic_reward *= self.exploration_degree[self.exploration_policy_idx]['beta']
                return intrinsic_reward
        else:
            return 0

    def extend_observations(self, decision_steps, terminal_steps):
        """
        Extend the current decision and terminal step through values from the prior step (action, rewards and expl.
        policy).

        Parameters
        ----------
        decision_steps:
            Contains the observation values, reward value and further information about the current step if not
            terminal.
        terminal_steps:
            Contains the observation values, reward value and further information about the current step if terminal.

        Returns
        -------
        decision_steps:
            Decision step but with extended observation values.
        terminal_steps:
            Terminal step but with extended observation values.
        """
        if len(decision_steps.obs[0]):
            if self.action_feedback:
                decision_steps.obs.append(np.array([self.prior_action], dtype=np.float32))
            if self.reward_feedback and self.intrinsic_exploration:
                decision_steps.obs.append(np.array([[self.prior_extrinsic_reward]], dtype=np.float32))
                decision_steps.obs.append(np.array([[self.prior_intrinsic_reward]], dtype=np.float32))
            if self.policy_feedback and self.intrinsic_exploration:
                decision_steps.obs.append(np.array([[self.exploration_policy_idx]], dtype=np.float32))
        else:
            if self.action_feedback:
                terminal_steps.obs.append(np.array([self.prior_action], dtype=np.float32))
            if self.reward_feedback and self.intrinsic_exploration:
                terminal_steps.obs.append(np.array([[self.prior_extrinsic_reward]], dtype=np.float32))
                terminal_steps.obs.append(np.array([[self.prior_intrinsic_reward]], dtype=np.float32))
            if self.policy_feedback and self.intrinsic_exploration:
                terminal_steps.obs.append(np.array([[self.exploration_policy_idx]], dtype=np.float32))
        return decision_steps, terminal_steps

    def update_prior_metrics(self, current_intrinsic_reward, actions, decision_steps, terminal_steps):
        """
        Save current step values for upcoming step.

        Parameters
        ----------
        current_intrinsic_reward:
            Reward calculated by exploration algorithm this step.
        actions:
            Actions used by agent this step.
        decision_steps:
            Contains the observation values, reward value and further information about the current step if not
            terminal.
        terminal_steps:
            Contains the observation values, reward value and further information about the current step if terminal.
        """
        self.prior_intrinsic_reward = current_intrinsic_reward
        if self.action_feedback and not len(terminal_steps.obs[0]):
            self.prior_action = actions[0]
        if self.reward_feedback and self.intrinsic_exploration:
            if not len(terminal_steps.obs[0]):
                self.prior_extrinsic_reward = decision_steps.reward[0]
            else:
                self.prior_extrinsic_reward = terminal_steps.reward[0]

    def get_exploration_policy_index(self):
        return self.exploration_policy_idx

    def get_exploration_reward(self):
        return self.prior_intrinsic_reward
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
