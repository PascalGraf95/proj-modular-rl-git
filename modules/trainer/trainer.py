#!/usr/bin/env python

from ..misc.replay_buffer import FIFOBuffer, PrioritizedBuffer
from ..misc.logger import GlobalLogger

import os
import numpy as np
from datetime import datetime

import yaml
import ray


class Trainer:
    """ Modular Reinforcement Learning© Trainer-class
    This base class contains all the relevant functions to acquire information from the environment and other modules
    so that agents can be constructed and algorithms can be adjusted accordingly.
    Furthermore, this class controls the training and testing loop and validates the given parameters.
    """

    def __init__(self):
        # Trainer Configuration
        self.config_key = ""
        self.trainer_configuration = None
        self.last_debug_message = 0
        self.logging_frequency = 100

        # Global Tensorboard Logger
        self.global_logger = None
        self.logging_name = None

        # Environment
        self.environment_configuration = None

        # Agent
        self.actors = []
        self.learner = None
        self.agent_configuration = None
        self.save_all_models = False
        self.remove_old_checkpoints = False

        # Exploration Algorithm
        self.exploration_configuration = None
        self.exploration_algorithm = None

        # Curriculum Learning Strategy
        self.global_curriculum_strategy = None

        # Preprocessing Algorithm
        self.preprocessing_algorithm = None

        # ReplayBuffer
        self.global_buffer = None

        # Training Loop
        # self.training_loop = self.async_training_loop()
        # self.testing_loop = self.async_testing_loop()

    # region Configuration Acquirement
    def get_agent_configuration(self):
        """Gather the parameters requested by the agent for the selected algorithm.

        :return: None
        """
        return Learner.get_config()
    # endregion

    # region Algorithm Choice
    def select_training_algorithm(self, training_algorithm):
        """ Imports the agent (actor, learner) according to the algorithm choice.

        :param training_algorithm: str
            Training algorithm name either "DQN", "DDPG", "TD3", "SAC"
        :return:
        """
        global Actor, Learner
        if training_algorithm == "DQN":
            from ..training_algorithms.DQN import DQNAgent as Agent
        elif training_algorithm == "DDPG":
            from ..training_algorithms.DDPG import DDPGAgent as Agent
        elif training_algorithm == "TD3":
            from ..training_algorithms.TD3 import TD3Agent as Agent
        elif training_algorithm == "SAC":
            from ..training_algorithms.SAC import SACActor as Actor
            from ..training_algorithms.SAC import SACLearner as Learner
        else:
            raise ValueError("There is no {} training algorithm.".format(training_algorithm))

    def select_curriculum_strategy(self, curriculum_strategy):
        """
        Imports the curriculum strategy according to the algorithm choice.
        :param curriculum_strategy: str
            Choose from None, "LinearCurriculum", "RememberingCurriculum" and "CrossFadeCurriculum"
        :return:
        """
        global CurriculumStrategy

        if curriculum_strategy == "None" or not curriculum_strategy:
            from ..curriculum_strategies.curriculum_strategy_blueprint import CurriculumStrategy
        elif curriculum_strategy == "LinearCurriculum":
            from ..curriculum_strategies.linear_curriculum import LinearCurriculum as CurriculumStrategy
        elif curriculum_strategy == "RememberingCurriculum":
            from ..curriculum_strategies.remembering_curriculum import RememberingCurriculum as CurriculumStrategy
        elif curriculum_strategy == "CrossFadeCurriculum":
            from ..curriculum_strategies.cross_fade_curriculum import CrossFadeCurriculum as CurriculumStrategy
        else:
            raise ValueError("There is no {} curriculum strategy.".format(curriculum_strategy))
    # endregion

    # region Validation
    def validate_trainer_configuration(self):
        """ Compare the content of the trainer configuration file with the parameter requests by the agent and the
        exploration algorithm.

        :return: Multiple lists with missing, obsolete and wrong type parameters. If all lists are empty, no
        warning or errors will occur.
        """
        return Learner.validate_config(self.trainer_configuration,
                                       self.agent_configuration,
                                       self.exploration_configuration)

    def validate_action_space(self):
        """Validate that the action spaces of the selected algorithm and the connected environment are compatible.

        :return: None
        """
        return Learner.validate_action_space(self.agent_configuration, self.environment_configuration)
    # endregion

    # region Instantiation
    def async_instantiate_agent(self, mode, interface: str, preprocessing_algorithm: str, exploration_algorithm: str,
                                environment_path: str = None, model_path: str = None,
                                preprocessing_path: str = None):
        """ Instantiate the agent consisting of learner and actor(s) and their respective submodules in an asynchronous
        fashion utilizing the ray library.

        :param exploration_algorithm: str
        :param interface: str
        :param preprocessing_algorithm: str
        :param preprocessing_path: str
        :param environment_path: str
        :param mode: str
        :param model_path: str
        :return:
        """
        # Initialize ray for parallel multiprocessing.
        ray.init()
        # If the connection is established directly with the Unity Editor or if we are in testing mode, override
        # the number of actors with 1.
        if not environment_path or mode == "testing":
            actor_num = 1
        else:
            actor_num = self.trainer_configuration.get("ActorNum")

        # If there is only one actor in training mode the degree of exploration should start at maximum. This only has
        # an effect if the exploration algorithm is not None. Otherwise each actor will be instantiated with a different
        # degree of exploration.
        if actor_num == 1 and mode == "training":
            exploration_degree = [1.0]
        else:
            exploration_degree = np.linspace(0, 1, actor_num)

        # Create actors and connect them to one environment each.
        self.actors = [Actor.remote(5004 + idx, mode,
                                    interface,
                                    preprocessing_algorithm,
                                    preprocessing_path,
                                    exploration_algorithm,
                                    environment_path,
                                    '/cpu:0') for idx in range(actor_num)]
        # Connect to either the Unity or Gym environment
        if interface == "MLAgentsV18":
            [actor.connect_to_unity_environment.remote() for actor in self.actors]
        else:
            [actor.connect_to_gym_environment.remote() for actor in self.actors]

        # For each actor instantiate the necessary modules
        for i, actor in enumerate(self.actors):
            actor.instantiate_modules.remote(self.trainer_configuration, exploration_degree[i])
            if mode == "training":
                actor.set_unity_parameters.remote(time_scale=1000, width=10, height=10, quality_level=1)
            else:
                actor.set_unity_parameters.remote(time_scale=1, width=500, height=500)

        # Get the environment configuration from the first actor's env
        environment_configuration = self.actors[0].get_environment_configuration.remote()
        self.environment_configuration = ray.get(environment_configuration)
        exploration_configuration = self.actors[0].get_exploration_configuration.remote()
        self.exploration_configuration = ray.get(exploration_configuration)
        agent_configuration = self.get_agent_configuration()
        # self.agent_configuration = ray.get(agent_configuration)

        # WARNING: Parameter mismatches will currently not be noticed due to a ray error with static methods
        # TODO: Fix this!
        '''
        # Check if parameters in trainer configuration match the requested algorithm parameters
        assert print_parameter_mismatches(*self.validate_trainer_configuration()), \
            "ERROR: Execution failed due to parameter mismatch."
        '''

        # Construct the learner
        self.learner = Learner.remote(mode, self.trainer_configuration,
                                      self.environment_configuration,
                                      self.trainer_configuration.get("NetworkParameters"),
                                      model_path)

        # Initialize the actor network for each actor
        network_ready = [actor.build_network.remote(self.trainer_configuration.get("NetworkParameters"),
                                                    self.environment_configuration, idx)
                         for idx, actor in enumerate(self.actors)]
        ray.get(network_ready)

        # Initialize the global buffer
        if self.trainer_configuration["PrioritizedReplay"]:
            self.global_buffer = PrioritizedBuffer.remote(capacity=self.trainer_configuration["ReplayCapacity"],
                                                          priority_alpha=self.trainer_configuration["PriorityAlpha"])
        else:
            self.global_buffer = FIFOBuffer.remote(capacity=self.trainer_configuration["ReplayCapacity"],
                                                   agent_num=self.environment_configuration["AgentNumber"],
                                                   n_steps=self.trainer_configuration.get("NSteps"),
                                                   gamma=self.trainer_configuration["Gamma"],
                                                   store_trajectories=False)
        if mode == "training":
            # Initialize the global curriculum strategy
            self.global_curriculum_strategy = CurriculumStrategy.remote()

        # Initialize the global logger
        self.logging_name = \
            datetime.strftime(datetime.now(), '%y%m%d_%H%M%S_') + self.trainer_configuration['TrainingID']
        self.global_logger = GlobalLogger.remote(os.path.join("training/summaries", self.logging_name),
                                                 actor_num=self.trainer_configuration["ActorNum"],
                                                 tensorboard=(mode == 'training'))
        if mode == 'training':
            if not os.path.isdir(os.path.join("./training/summaries", self.logging_name)):
                os.makedirs(os.path.join("./training/summaries", self.logging_name))
            with open(os.path.join("./training/summaries", self.logging_name, "training_parameters.yaml"), 'w') as file:
                _ = yaml.dump(self.trainer_configuration, file)
                _ = yaml.dump(self.environment_configuration, file)
                _ = yaml.dump(self.exploration_configuration, file)
    # endregion

    # region Misc
    def get_elapsed_training_time(self):
        """

        :return:
        """
        return self.global_logger.get_elapsed_time.remote()

    def async_save_agent_models(self, training_step):
        """

        :param training_step:
        :return:
        """
        checkpoint_condition = self.global_logger.check_checkpoint_condition.remote()
        self.learner.save_checkpoint.remote(os.path.join("training/summaries", self.logging_name),
                                            self.global_logger.get_best_running_average.remote(),
                                            training_step, self.save_all_models, checkpoint_condition)

    def boost_exploration(self):
        """

        :return:
        """
        if not self.learner.boost_exploration():
            self.exploration_algorithm.boost_exploration()

    def parse_training_parameters(self, parameter_path, parameter_dict=None):
        """

        :param parameter_path:
        :param parameter_dict:
        :return:
        """
        with open(parameter_path) as file:
            trainer_configuration = yaml.safe_load(file)
        if parameter_dict:
            self.trainer_configuration = trainer_configuration[parameter_dict]
        else:
            return [key for key, val in trainer_configuration.items()]
        return self.trainer_configuration
    # endregion

    # region Environment Interaction
    def async_training_loop(self):
        """

        :return:
        """
        # region Initialization
        # Get the task properties from one of the environments
        unity_responded, task_properties = self.actors[0].get_task_properties.remote(
            self.global_curriculum_strategy.has_unity_responded.remote())
        self.global_curriculum_strategy.update_task_properties.remote(unity_responded, task_properties)

        # Synchronise all actor networks with the learner networks
        network_update_requested = self.learner.is_network_update_requested.remote()
        [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(network_update_requested))
         for actor in self.actors]
        # endregion

        while True:
            # Each actor plays one step in the environment
            actors_ready = [actor.play_one_step.remote() for actor in self.actors]

            # region Curriculum Info Acquisition
            # If the task level changed try to get new task information from the environment
            for actor in self.actors:
                unity_responded_global = self.global_curriculum_strategy.has_unity_responded.remote()
                unity_responded, task_properties = actor.get_task_properties.remote(unity_responded_global)
                # If the retrieved information match the target level information update the global curriculum
                self.global_curriculum_strategy.update_task_properties.remote(unity_responded, task_properties)
            # endregion
            
            for idx, actor in enumerate(self.actors):
                # If an actor has collected enough samples, copy the samples from its local buffer to the global buffer.
                # In case of an Prioritized Experience Replay let the actor calculate an initial priority.
                # Also make sure the environment runs the desired curriculum level
                actor.send_target_task_level.remote()
                samples, indices = actor.get_new_samples.remote()
                if self.trainer_configuration["PrioritizedReplay"]:
                    sample_errors = actor.get_sample_errors.remote(samples)
                    self.global_buffer.append_list.remote(samples, sample_errors)
                else:
                    self.global_buffer.append_list.remote(samples)
                self.global_logger.append.remote(*actor.get_new_stats.remote(), actor_idx=idx)

            # Check if enough new samples have been collected and the minimum capacity is reached in order to
            # start a training cycle.
            # Sample a new batch of transitions from the global replay buffer
            samples, indices = self.global_buffer.sample.remote(self.trainer_configuration,
                                                                self.trainer_configuration.get("BatchSize"))
            # Train the learner with the batch and observer the resulting metrics
            training_metrics, sample_errors, training_step = self.learner.learn.remote(samples)
            # Update the actor networks if requested by the learner
            network_update_requested = self.learner.is_network_update_requested.remote()
            [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(network_update_requested))
             for actor in self.actors]

            # Update the prioritized experience replay buffer with the td-errors
            self.global_buffer.update.remote(indices, sample_errors)
            # Train the actors exploration algorithm with the same batch
            [actor.exploration_learning_step.remote(samples) for actor in self.actors]
            # Check if either the reward threshold for a level change has been reached or if the curriculum
            # strategy transitions between different levels by default
            _, max_reward, total_episodes = self.global_logger.get_current_max_stats.remote(
                self.global_curriculum_strategy.get_average_episodes.remote())

            # if self.global_curriculum_strategy.check_task_level_change_condition.remote(max_reward, total_episodes) or \
            #         self.global_curriculum_strategy.get_level_transition.remote():
            task_level_condition = \
                self.global_curriculum_strategy.check_task_level_change_condition.remote(max_reward,
                                                                                         total_episodes)
            # TODO Note: Currently this only works for the Linear Curriculum
            self.global_logger.register_level_change.remote(task_level_condition)
            target_task_level = self.global_curriculum_strategy.get_new_task_level.remote(task_level_condition)
            for actor in self.actors:
                actor.set_new_target_task_level.remote(target_task_level)

            # Check if a new best reward has been achieved, if so save the models
            self.async_save_agent_models(training_step)
            if self.remove_old_checkpoints:
                self.global_logger.remove_old_checkpoints()

            # Log training results to Tensorboard
            self.global_logger.log_dict.remote(training_metrics, training_step, self.logging_frequency)
            for idx, actor in enumerate(self.actors):
                self.global_logger.log_dict.remote(actor.get_exploration_logs.remote(idx), training_step,
                                                   self.logging_frequency)

            # Get the mean episode length + reward from the best performing actor
            # mean_episode_length, mean_episode_reward, episodes = self.global_logger.get_current_max_stats(
            #     self.global_curriculum_strategy.average_episodes)
            ray.wait(actors_ready)

    def async_testing_loop(self):
        """

        :return:
        """
        # Synchronise all actor networks with the learner networks
        [actor.update_actor_network.remote(ray.get(self.learner.get_actor_network_weights.remote()))
         for actor in self.actors]

        while True:
            # Each actor plays one step in the environment
            [actor.play_one_step.remote() for actor in self.actors]

            # If an actor has collected enough samples, copy the samples from its local buffer to the global buffer.
            # In case of an Prioritized Experience Replay let the actor calculate an initial priority.
            for idx, actor in enumerate(self.actors):
                if ray.get(actor.is_minimum_capacity_reached.remote()):
                    samples = ray.get(actor.get_new_samples.remote())[0]
                    if self.trainer_configuration["PrioritizedReplay"]:
                        sample_errors = ray.get(actor.get_sample_errors.remote(samples))
                        self.global_buffer.append_list.remote(samples, sample_errors)
                    else:
                        self.global_buffer.append_list.remote(samples)
                    lengths, rewards, total_episodes_played = ray.get(actor.get_new_stats.remote())
                    if lengths:
                        self.global_logger.append(lengths, rewards, total_episodes_played, actor_idx=idx)

            # Get the mean episode length + reward from the best performing actor
            mean_episode_length, mean_episode_reward, episodes = self.global_logger.get_current_max_stats(10)

            yield mean_episode_length, mean_episode_reward, episodes, 0, \
                0, [0]*4
    # endregion


def print_parameter_mismatches(mis_par, obs_par, mis_net_par, obs_net_par, mis_expl_par, obs_expl_par,
                               wro_type, wro_type_net, wro_type_expl):
    """Prints the missing and obsolete parameters from the trainer configuration file for the agent, exploration
    algorithm and network construction.

    :param mis_par: Parameters that are requested by the agent but missing in the trainer configuration file.
    :param obs_par: Parameters that in the trainer configuration file.
    :param mis_net_par: Parameters that are requested for the network construction
    but missing in the trainer configuration file .
    :param obs_net_par: Parameters that in the trainer configuration file.
    :param mis_expl_par: Parameters that are requested by the agent but missing in the trainer configuration file.
    :param obs_expl_par: Parameters that in the trainer configuration file.
    :param wro_type: Parameters that are of the wrong data type in the trainer configuration file.
    :param wro_type_net: Parameters that are of the wrong data type in the trainer configuration file.
    :param wro_type_expl: Parameters that are of the wrong data type in the trainer configuration file.
    :return:
    """
    successful = True
    if len(mis_par):
        successful = False
        print("ERROR: The following parameters which are requested by the agent "
              "are not defined in the trainer configuration: {}".format(", ".join(mis_par)))
    if len(obs_par):
        print("WARNING: The following parameters are not requested by the agent "
              "but are defined in the trainer configuration: {}".format(", ".join(obs_par)))

    if len(mis_net_par):
        successful = False
        print("ERROR: The following network parameters which are requested by the agent "
              "are not defined in the trainer configuration: {}".format(", ".join(mis_net_par)))
    if len(obs_net_par):
        print("WARNING: The following network parameters are not requested by the agent "
              "but are defined in the trainer configuration: {}".format(", ".join(obs_net_par)))

    if len(mis_expl_par):
        successful = False
        print("ERROR: The following exploration parameters which are requested by the algorithm "
              "are not defined in the trainer configuration: {}".format(", ".join(mis_expl_par)))
    if len(obs_expl_par):
        print("WARNING: The following exploration parameters are not requested by the algorithm "
              "but are defined in the trainer configuration: {}".format(", ".join(obs_expl_par)))

    if len(wro_type):
        successful = False
        print("ERROR: The following training parameters do not match the "
              "data types requested by the agent: {}".format(", ".join(wro_type)))

    if len(wro_type_net):
        successful = False
        print("ERROR: The following network parameters do not match the data types requested by "
              "the agent network: {}".format(", ".join(wro_type_net)))

    if len(wro_type_net):
        successful = False
        print("ERROR: The following exploration parameters do not match the data types requested by "
              "the algorithm: {}".format(", ".join(wro_type_expl)))
    return successful
