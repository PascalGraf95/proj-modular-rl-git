#!/usr/bin/env python

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from ..sidechannel.curriculum_sidechannel import CurriculumSideChannelTaskInfo
from ..misc.replay_buffer import FIFOBuffer
from ..misc.logger import GlobalLogger
import gym
import tensorflow as tf
import time


import os
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

import yaml
from typing import Any, Dict, TextIO

import collections
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

        # Logger
        self.global_logger = None
        self.logging_name = None

        # Environment
        self.env = None
        self.engine_configuration_channel = None
        self.environment_configuration = None
        self.environment_selection = None

        # Interface
        self.connect = None

        # Agent
        self.actors = []
        self.learner = None
        self.agent_configuration = None
        self.behavior_name = None
        self.model_path = None
        self.save_all_models = False
        self.remove_old_checkpoints = False

        # Exploration Algorithm
        self.exploration_configuration = None
        self.exploration_algorithm = None

        # Curriculum Learning Strategy
        self.global_curriculum_strategy = None
        self.curriculum_channel_task_info = None

        # Preprocessing Algorithm
        self.preprocessing_algorithm = None

        # ReplayBuffer
        self.global_buffer = None

        # Training Loop
        self.logging_frequency = 100
        self.training_loop = self.training_loop_generator()
        self.testing_loop = self.testing_loop_generator()
        self.actor_playing_generator = None

        # Testing
        self.test_frequency = 0
        self.test_episodes = 10
        self.played_test_episodes = 0
        self.test_flag = False

        # Misc
        self.last_debug_message = 0

    # region Configuration Acquirement
    def get_agent_configuration(self):
        """Gather the parameters requested by the agent for the selected algorithm.

        :return: None
        """
        return Learner.get_config()
    # endregion

    # region Algorithm Choice
    def select_training_algorithm(self, training_algorithm):
        """ Imports the agent according to the algorithm choice.

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
        global CurriculumStrategy

        if curriculum_strategy == "None":
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
    def instantiate_agent(self, mode, interface, preprocessing_algorithm, exploration_algorithm,
                          environment_path=None, model_path=None, preprocessing_path=None):
        """

        :param exploration_algorithm:
        :param interface:
        :param preprocessing_algorithm:
        :param preprocessing_path:
        :param environment_path:
        :param mode: str
        :param model_path: str
        :return:
        """
        # Initialize ray for parallel multiprocessing
        ray.init()
        # If the connection is established directly with the Unity Editor or if we are in testing mode, override
        # the number of actors with 1.
        if not environment_path or mode == "testing":
            actor_num = 1
            exploration_degree = [1.0]
        else:
            actor_num = self.trainer_configuration.get("ActorNum")
            exploration_degree = np.linspace(0, 1, actor_num)

        # Create actors and connect them to one environment each, also instantiate all the modules
        self.actors = [Actor.remote(5004 + i, mode,
                                    interface,
                                    preprocessing_algorithm,
                                    preprocessing_path,
                                    exploration_algorithm,
                                    environment_path) for i in range(actor_num)]
        # The connect function is currently hard-coded to connect to unity because of assignment errors with ray!
        connected = [actor.connect_to_unity_environment.remote() for actor in self.actors]

        # For each environment instantiate necessary modules
        for i, actor in enumerate(self.actors):
            actor.instantiate_modules.remote(self.trainer_configuration, exploration_degree[i])
            if mode == "training":
                actor.set_unity_parameters.remote(time_scale=1000, width=10, height=10, quality_level=1)
            else:
                actor.set_unity_parameters.remote(time_scale=1)

        # Get the environment configuration from the first actor's env
        environment_configuration = self.actors[0].get_environment_configuration.remote()
        self.environment_configuration = ray.get(environment_configuration)
        exploration_configuration = self.actors[0].get_exploration_configuration.remote()
        self.exploration_configuration = ray.get(exploration_configuration)
        agent_configuration = self.get_agent_configuration()
        # self.agent_configuration = ray.get(agent_configuration)

        # WARNING: Parameter mismatches will currently not be noticed due to a ray error with static methods
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

        # Give the initial actor network to each actor
        for actor in self.actors:
            actor.update_actor_network.remote(self.learner.get_actor_network.remote())

        # Initialize the global buffer
        self.global_buffer = FIFOBuffer(capacity=self.trainer_configuration["ReplayCapacity"],
                                        agent_num=self.environment_configuration["AgentNumber"],
                                        n_steps=self.trainer_configuration.get("NSteps"),
                                        gamma=self.trainer_configuration["Gamma"],
                                        store_trajectories=False,
                                        prioritized=self.trainer_configuration["PrioritizedReplay"])

        # Initialize the global curriculum strategy
        self.global_curriculum_strategy = CurriculumStrategy()

        # Initialize the global logger
        self.logging_name = \
            datetime.strftime(datetime.now(), '%y%m%d_%H%M%S_') + self.trainer_configuration['TrainingID']
        self.global_logger = GlobalLogger(os.path.join("training/summaries", self.logging_name),
                                          actor_num=self.trainer_configuration["ActorNum"],
                                          tensorboard=(mode == 'training'))
        if mode == 'training':
            with open(os.path.join("./training/summaries", self.logging_name, "training_parameters.yaml"), 'w') as file:
                _ = yaml.dump(self.trainer_configuration, file)
                _ = yaml.dump(self.environment_configuration, file)
                _ = yaml.dump(self.exploration_configuration, file)

        print("INFO: Initialization successful!")
    # endregion

    # region Misc
    def get_elapsed_training_time(self):
        """

        :return:
        """
        return self.global_logger.get_elapsed_time()

    def save_agent_models(self, training_step):
        """

        :param training_step:
        :return:
        """
        self.learner.save_checkpoint(os.path.join("training/summaries", self.logging_name),
                                     self.global_logger.best_running_average_reward,
                                     training_step, save_all_models=self.save_all_models)

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
    def training_loop_generator(self):
        """

        :return:
        """
        training_step = 0
        training_metrics = {}
        # Update to the current task properties
        self.global_curriculum_strategy.update_task_properties(*ray.get(self.actors[0].get_task_properties.remote()))

        while True:
            for actor in self.actors:
                # Play the next step in each episode
                next(actor.playing_generator)
                # If the task level changed try to get new task information from the environment
                if not self.global_curriculum_strategy.unity_responded:
                    unity_responded, task_properties = actor.get_task_properties()
                    # If the retrieved information match the target level information update the global curriculum
                    if task_properties[1] == actor.target_task_level:
                        self.global_curriculum_strategy.update_task_properties(unity_responded, task_properties)

            for idx, actor in enumerate(self.actors):
                if actor.network_update_requested:
                    actor.update_actor_network(self.learner.get_actor_network())

                if actor.minimum_capacity_reached:
                    self.global_buffer.append_list(actor.get_new_samples()[0])
                    self.global_logger.append(*actor.get_new_stats(), actor_idx=idx)

            # Check if enough new samples have been collected and the minimum capacity is reached
            if self.global_buffer.check_training_condition(self.trainer_configuration):
                # Sample a new batch of transitions from the global replay buffer
                samples, indices = self.global_buffer.sample(self.trainer_configuration.get("BatchSize"))
                # Train the learner with the batch
                training_metrics, sample_errors, training_step = self.learner.learn(samples)
                # Train the actors exploration algorithm with the same batch
                for actor in self.actors:
                    actor.exploration_learning_step(samples)
                # Check if either the reward threshold for a level change has been reached or if the curriculum
                # strategy transitions between different levels
                if self.global_curriculum_strategy.check_task_level_change_condition(
                        *self.global_logger.get_current_max_stats(self.global_curriculum_strategy.average_episodes)) or \
                        self.global_curriculum_strategy.level_transition:
                    target_task_level = self.global_curriculum_strategy.get_new_task_level()
                    for actor in self.actors:
                        actor.set_new_target_task_level(target_task_level)
                # Check if a new best reward has been achieved and save the models
                if self.global_logger.check_checkpoint_condition():
                    self.save_agent_models(training_step)
                    if self.remove_old_checkpoints:
                        self.global_logger.remove_old_checkpoints()

                # Log training results to Tensorboard and console
                if training_step % self.logging_frequency == 0:
                    self.global_logger.log_dict(training_metrics, training_step)
                    self.global_logger.log_dict(self.actors[0].exploration_algorithm.get_logs(), training_step)

            # Get the mean episode length + reward from the best performing actor
            mean_episode_length, mean_episode_reward, episodes = self.global_logger.get_episode_stats()

            yield mean_episode_length, mean_episode_reward, episodes, training_step, \
                training_metrics.get("Losses/Loss"), self.global_curriculum_strategy.return_task_properties()

    def testing_loop_generator(self):
        """

        :return:
        """
        while True:
            for actor in self.actors:
                # Play the next step in each episode
                next(actor.playing_generator)
                # If the task level changed try to get new task information from the environment

            for actor in self.actors:
                if actor.network_update_requested:
                    actor.update_actor_network(self.learner.get_actor_network())

                if actor.minimum_capacity_reached:
                    self.global_buffer.append_list(actor.get_new_samples()[0])
                    self.global_logger.append(*actor.get_new_stats())

            if self.global_logger.new_episodes:
                # Get the mean episode length + reward from the best performing actor
                mean_episode_length, mean_episode_reward, episodes = self.global_logger.get_episode_stats()

                yield mean_episode_length, mean_episode_reward, episodes, 0, 0, \
                      self.global_curriculum_strategy.return_task_properties()
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
