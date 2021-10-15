#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from ..misc.logger import LocalLogger
from ..misc.replay_buffer import LocalFIFOBuffer, LocalRecurrentBuffer
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from ..sidechannel.curriculum_sidechannel import CurriculumSideChannelTaskInfo
from ..curriculum_strategies.curriculum_strategy_blueprint import CurriculumCommunicator
from mlagents_envs.environment import UnityEnvironment, ActionTuple
import time
import ray


class Actor:
    def __init__(self, port: int, mode: str,
                 interface: str,
                 preprocessing_algorithm: str,
                 preprocessing_path: str,
                 exploration_algorithm: str,
                 environment_path: str = "",
                 device: str = '/cpu:0'):
        # Network
        self.actor_network = None
        self.critic_network = None
        self.actor_prediction_network = None
        self.network_update_requested = False
        self.new_steps_taken = 0
        self.network_update_frequency = 1000

        # Environment
        self.environment = None
        self.environment_configuration = None
        self.environment_path = environment_path

        # Local Buffer
        self.local_buffer = None
        self.minimum_capacity_reached = False

        # Recurrent Properties
        self.recurrent = None
        self.sequence_length = None

        # Local Logger
        self.local_logger = None

        # Tensorflow Device
        self.device = device

        # Behavior Parameters
        self.behavior_name = None
        self.action_shape = None
        self.action_type = None
        self.observation_shapes = None
        self.agent_number = None

        # Exploration Algorithm
        self.exploration_configuration = None
        self.exploration_algorithm = None
        self.adaptive_exploration = False

        # Curriculum Learning Strategy & Engine Side Channel
        self.engine_configuration_channel = None
        self.curriculum_communicator = None
        self.curriculum_side_channel = None
        self.target_task_level = 0

        # Preprocessing Algorithm
        self.preprocessing_algorithm = None
        self.preprocessing_path = preprocessing_path

        # Algorithm Selection
        self.select_agent_interface(interface)
        self.select_exploration_algorithm(exploration_algorithm)
        self.select_preprocessing_algorithm(preprocessing_algorithm)

        # Prediction Parameters
        self.gamma = None
        self.n_steps = None

        # Mode
        self.mode = mode
        self.port = port

    # region Environment Connection
    def connect(self):
        return

    def connect_to_unity_environment(self):
        self.engine_configuration_channel = EngineConfigurationChannel()
        self.curriculum_side_channel = CurriculumSideChannelTaskInfo()
        self.environment = UnityEnvironment(file_name=self.environment_path,
                                            side_channels=[self.engine_configuration_channel,
                                                           self.curriculum_side_channel], base_port=self.port)
        self.environment.reset()
        return True

    def connect_to_gym_environment(self):
        self.environment = AgentInterface.connect(self.environment_path)

    def set_unity_parameters(self, **kwargs):
        self.engine_configuration_channel.set_configuration_parameters(**kwargs)
    # endregion

    # region Property Query
    def is_minimum_capacity_reached(self):
        return self.minimum_capacity_reached

    def get_target_task_level(self):
        return self.target_task_level

    @ray.method(num_returns=2)
    def get_task_properties(self, unity_responded_global):
        if not unity_responded_global:
            unity_responded, task_properties = self.curriculum_communicator.get_task_properties()
            return unity_responded, task_properties
        else:
            return None, None

    def get_exploration_logs(self, idx):
        return self.exploration_algorithm.get_logs(idx)

    def get_environment_configuration(self):
        return self.environment_configuration

    def read_environment_configuration(self):
        self.behavior_name = AgentInterface.get_behavior_name(self.environment)
        self.action_shape = AgentInterface.get_action_shape(self.environment)
        self.action_type = AgentInterface.get_action_type(self.environment)
        self.observation_shapes = AgentInterface.get_observation_shapes(self.environment)
        self.agent_number = AgentInterface.get_agent_number(self.environment, self.behavior_name)

        self.environment_configuration = {"BehaviorName": self.behavior_name,
                                          "ActionShape": self.action_shape,
                                          "ActionType": self.action_type,
                                          "ObservationShapes": self.observation_shapes,
                                          "AgentNumber": self.agent_number}

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
            from ..interfaces.mlagents_v18 import MlAgentsV18Interface as AgentInterface
            self.connect = self.connect_to_unity_environment

        elif interface == "OpenAIGym":
            from ..interfaces.openaigym import OpenAIGymInterface as AgentInterface
            self.connect = self.connect_to_gym_environment
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
            from ..exploration_algorithms.intrinsic_curiosity_module import IntrinsicCuriosityModule as ExplorationAlgorithm
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
        """
        self.local_buffer = LocalFIFOBuffer(capacity=5000,
                                            agent_num=self.agent_number,
                                            n_steps=trainer_configuration.get("NSteps"),
                                            gamma=trainer_configuration.get("Gamma"),
                                            store_trajectories=False)
        """
        if self.recurrent:
            self.local_buffer = LocalRecurrentBuffer(capacity=5000,
                                                     agent_num=self.agent_number,
                                                     n_steps=trainer_configuration.get("NSteps"),
                                                     sequence_length=trainer_configuration.get("SequenceLength"),
                                                     gamma=trainer_configuration.get("Gamma"))
        else:
            self.local_buffer = LocalFIFOBuffer(capacity=5000,
                                                agent_num=self.agent_number,
                                                n_steps=trainer_configuration.get("NSteps"),
                                                gamma=trainer_configuration.get("Gamma"))
        self.network_update_frequency = trainer_configuration.get("NetworkUpdateFrequency")

    def instantiate_modules(self, trainer_configuration, exploration_degree):
        self.recurrent = trainer_configuration["Recurrent"]
        self.sequence_length = trainer_configuration.get("SequenceLength")
        self.gamma = trainer_configuration["Gamma"]
        self.n_steps = trainer_configuration["NSteps"]
        self.read_environment_configuration()
        self.adaptive_exploration = trainer_configuration.get("AdaptiveExploration")
        trainer_configuration["ExplorationParameters"]["ExplorationDegree"] = exploration_degree
        self.exploration_algorithm = ExplorationAlgorithm(self.environment_configuration["ActionShape"],
                                                          self.environment_configuration["ObservationShapes"],
                                                          self.environment_configuration["ActionType"],
                                                          trainer_configuration["ExplorationParameters"])
        self.curriculum_communicator = CurriculumCommunicator(self.curriculum_side_channel)

        self.preprocessing_algorithm = PreprocessingAlgorithm(self.preprocessing_path)
        modified_output_shapes = self.preprocessing_algorithm.get_output_shapes(self.environment_configuration)
        self.environment_configuration["ObservationShapes"] = modified_output_shapes
        self.observation_shapes = modified_output_shapes

        self.local_logger = LocalLogger(agent_num=self.agent_number)
        self.instantiate_local_buffer(trainer_configuration)
    # endregion

    # region Network Construction and Updates
    def build_network(self, network_parameters, environment_parameters, idx):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def update_actor_network(self, network_weights):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def reset_actor_state(self):
        """This function resets the actors LSTM states to random values at the beginning of a new episode."""
        for layer in self.actor_network.layers:
            if "lstm" in layer.name:
                layer.reset_states()
    # endregion

    # region Environment Interaction
    def play_one_step(self):
        # Step acquisition (steps contain states, done_flags and rewards)
        decision_steps, terminal_steps = AgentInterface.get_steps(self.environment, self.behavior_name)
        # Preprocess steps if an respective algorithm has been activated
        decision_steps, terminal_steps = self.preprocessing_algorithm.preprocess_observations(decision_steps,
                                                                                              terminal_steps)
        # Choose the next action either by exploring or exploiting
        actions = self.exploration_algorithm.act(decision_steps)
        if actions is None:
            actions = self.act(decision_steps.obs, mode=self.mode)

        # Append steps and actions to the local replay buffer
        self.local_buffer.add_new_steps(terminal_steps.obs, terminal_steps.reward,
                                        terminal_steps.agent_id,
                                        step_type="terminal")
        self.local_buffer.add_new_steps(decision_steps.obs, decision_steps.reward,
                                        decision_steps.agent_id,
                                        actions=actions, step_type="decision")

        # Track the rewards in a local logger
        self.local_logger.track_episode(terminal_steps.reward, terminal_steps.agent_id,
                                        step_type="terminal")
        self.local_logger.track_episode(decision_steps.reward, decision_steps.agent_id,
                                        step_type="decision")

        # If enough samples have been collected, mark local buffer ready for readout
        if len(self.local_buffer) >= 10:
            self.minimum_capacity_reached = True

        # If enough steps have been taken, mark agent ready for updated network
        # self.new_steps_taken += len(decision_steps)
        # if self.new_steps_taken >= self.network_update_frequency:
        #     self.network_update_requested = True

        # If all agents are in a terminal state reset the environment
        if self.local_buffer.check_reset_condition():
            AgentInterface.reset(self.environment)
            self.local_buffer.done_agents.clear()
            self.reset_actor_state()
        # Otherwise take a step in the environment according to the chosen action
        else:
            AgentInterface.step_action(self.environment, self.behavior_name, actions)

    def act(self, states, mode="training"):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Misc
    def exploration_learning_step(self, samples):
        if not samples:
            return
        if self.adaptive_exploration:
            self.exploration_algorithm.learning_step(samples)

    @ray.method(num_returns=2)
    def get_new_samples(self):
        if self.minimum_capacity_reached:
            self.minimum_capacity_reached = False
            return self.local_buffer.sample(-1, reset_buffer=True, random_samples=False)
        else:
            return None, None

    @ray.method(num_returns=3)
    def get_new_stats(self):
        return self.local_logger.get_episode_stats()

    def set_new_target_task_level(self, target_task_level):
        if target_task_level:
            self.target_task_level = target_task_level
            self.curriculum_side_channel.unity_responded = False

    def send_target_task_level(self):
        self.curriculum_communicator.set_task_number(self.target_task_level)

    def get_sample_errors(self, samples):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion


class Learner:
    # region ParameterSpace
    TrainingParameterSpace = {
        'ActorNum': int,
        'NetworkUpdateFrequency': int,
        'TrainingID': str,
        'BatchSize': int,
        'Gamma': float,
        'TrainingInterval': int,
        'NetworkParameters': list,
        'ExplorationParameters': dict,
        'ReplayMinSize': int,
        'ReplayCapacity': int,
        'NSteps': int,
        'SyncMode': str,
        'SyncSteps': int,
        'Tau': float,
        'ClipGrad': float,
        'PrioritizedReplay': bool,
        'AdaptiveExploration': bool
    }
    ActionType = []
    NetworkTypes = []
    Metrics = []

    def __init__(self, trainer_configuration, environment_configuration):
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

        # Recurrent Paramters
        self.recurrent = trainer_configuration.get('Recurrent')
        self.sequence_length = trainer_configuration.get('SequenceLength')

        # Misc
        self.training_step = 0
        self.steps_since_actor_update = 0
        self.set_gpu_growth()  # Important step to avoid tensorflow OOM errors when running multiprocessing!

    # endregion

    # region Network Construction and Transfer
    def is_network_update_requested(self):
        return self.steps_since_actor_update >= self.network_update_frequency

    def get_actor_network_weights(self):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def build_network(self, network_parameters, environment_parameters):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def set_gpu_growth(self):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

    def sync_models(self):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Checkpoints
    def load_checkpoint(self, path):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")

    def save_checkpoint(self, path, running_average_reward, training_step, save_all_models=False):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Learning
    def learn(self, replay_batch):
        raise NotImplementedError("Please overwrite this method in your algorithm implementation.")
    # endregion

    # region Misc
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
                print(transition)
                continue
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
    # endregion

    # region Parameter Validation
    @staticmethod
    def check_mismatching_parameters(test_configs, blueprint_configs):
        if type(test_configs) == list:
            missing_parameters = []
            obsolete_parameters = []
            for idx, (test_config, blueprint_config) in enumerate(zip(test_configs, blueprint_configs)):
                missing_parameters += [key + "({})".format(idx) for key, val
                                       in blueprint_config.items() if key not in test_config]
                obsolete_parameters += [key + "({})".format(idx) for key, val
                                        in test_config.items() if key not in blueprint_config]
        else:
            missing_parameters = [key for key, val in blueprint_configs.items() if key not in test_configs]
            obsolete_parameters = [key for key, val in test_configs.items() if key not in blueprint_configs]
        return missing_parameters, obsolete_parameters

    @staticmethod
    def validate_config(trainer_configuration,
                        agent_configuration,
                        exploration_configuration):
        missing_parameters, obsolete_parameters = \
            Learner.check_mismatching_parameters(trainer_configuration,
                                                 agent_configuration.get('TrainingParameterSpace'))

        missing_net_parameters, obsolete_net_parameters = \
            Learner.check_mismatching_parameters(trainer_configuration.get('NetworkParameters'),
                                                 agent_configuration.get('NetworkParameterSpace'))
        missing_expl_parameters, obsolete_expl_parameters = \
            Learner.check_mismatching_parameters(trainer_configuration.get('ExplorationParameters'),
                                                 exploration_configuration.get('ParameterSpace'))

        wrong_type_parameters = \
            Learner.check_wrong_parameter_types(trainer_configuration,
                                                agent_configuration.get('TrainingParameterSpace'),
                                                obsolete_parameters)

        wrong_type_net_parameters = \
            Learner.check_wrong_parameter_types(trainer_configuration.get('NetworkParameters')[0],
                                                agent_configuration.get('NetworkParameterSpace')[0],
                                                obsolete_net_parameters)
        wrong_type_expl_parameters = \
            Learner.check_wrong_parameter_types(trainer_configuration.get('ExplorationParameters'),
                                                exploration_configuration.get('ParameterSpace'),
                                                obsolete_expl_parameters)

        return missing_parameters, obsolete_parameters, missing_net_parameters, obsolete_net_parameters, \
            missing_expl_parameters, obsolete_expl_parameters, wrong_type_parameters, wrong_type_net_parameters, \
            wrong_type_expl_parameters

    @staticmethod
    def validate_action_space(agent_configuration, environment_configuration):
        # Check for compatibility of environment and agent action space
        if environment_configuration.get("ActionType") not in agent_configuration.get("ActionType"):
            print("The action spaces of the environment and the agent are not compatible.")
            return False
        return True

    @staticmethod
    def check_wrong_parameter_types(test_config, blueprint_config, obsolete=[]):
        wrong_type_parameters = [" >> ".join([": ".join([key, str(val)]), str(blueprint_config.get(key))])
                                 for key, val in test_config.items()
                                 if type(val) != blueprint_config.get(key)
                                 and key not in obsolete]
        return wrong_type_parameters
    # endregion











