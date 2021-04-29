#!/usr/bin/env python

"""
The Trainer connects an Unity environment to the learning python agent utilizing the Unity Python API.
Furthermore, it initiates and manages the training procedure.

Created by Pascal Graf
Last edited 09.04.2021
"""

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from ..sidechannel.curriculum_sidechannel import CurriculumSideChannelTaskInfo
from ..misc.replay_buffer import ReplayBuffer
from ..misc.logger import Logger
import gym
import tensorflow as tf


import os
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

import yaml
from typing import Any, Dict, TextIO

import collections


class Trainer:
    def __init__(self):
        # Config
        self.config_key = ""
        self.trainer_configuration = None
        self.logging_name = None
        # Environment
        self.env = None
        self.engine_configuration_channel = None
        self.environment_configuration = None
        self.environment_selection = None
        # Interface
        self.connect = None
        # Agent
        self.agent = None
        self.agent_configuration = None
        self.behavior_name = None
        self.model_path = None
        self.save_all_models = False
        self.remove_old_checkpoints = False
        # Exploration Algorithm
        self.exploration_configuration = None
        self.exploration_algorithm = None
        # Curriculum Learning Strategy
        self.curriculum_strategy = None
        self.last_debug_message = 0
        # Initiation Progress
        self.initiation_progress = 0
        # Logger
        self.logger = None
        # ReplayBuffer
        self.replay_buffer = None
        self.prediction_error_based = False
        # Training Loop
        self.training_loop = self.training_loop_generator()
        self.testing_loop = self.testing_loop_generator()
        # Testing
        self.test_frequency = 0
        self.test_episodes = 10
        self.played_test_episodes = 0
        self.test_flag = False
        # Curriculum SideChannel
        self.curriculum_channel_task_info = None

    def reset(self):
        self.__init__()

    def connect_to_unity_environment(self, environment_path=None):
        self.engine_configuration_channel = EngineConfigurationChannel()
        self.curriculum_channel_task_info = CurriculumSideChannelTaskInfo()
        self.curriculum_strategy.register_side_channels(self.curriculum_channel_task_info)
        self.env = UnityEnvironment(file_name=environment_path, base_port=5004,
                                    side_channels=[self.engine_configuration_channel,
                                                   self.curriculum_channel_task_info])
        self.env.reset()

    def connect_to_gym_environment(self):
        self.env = gym.make(self.environment_selection)
        self.env.reset()

    def set_unity_parameters(self, **kwargs):
        self.engine_configuration_channel.set_configuration_parameters(**kwargs)

    def get_environment_configuration(self):
        assert self.env, "The trainer has not been connected to an Unity environment yet."
        self.behavior_name = AgentInterface.get_behavior_name(self.env)
        action_shape = AgentInterface.get_action_shape(self.env)
        action_type = AgentInterface.get_action_type(self.env)
        observation_shapes = AgentInterface.get_observation_shapes(self.env)
        agent_number = AgentInterface.get_agent_number(self.env, self.behavior_name)

        self.environment_configuration = {"BehaviorName": self.behavior_name,
                                          "ActionShape": action_shape,
                                          "ActionType": action_type,
                                          "ObservationShapes": observation_shapes,
                                          "AgentNumber": agent_number}
        return self.environment_configuration

    def parse_training_parameters(self, parameter_path, parameter_dict=None):
        with open(parameter_path) as file:
            trainer_configuration = yaml.safe_load(file)
        if parameter_dict:
            self.trainer_configuration = trainer_configuration[parameter_dict]
        else:
            return [key for key, val in trainer_configuration.items()]
        return self.trainer_configuration

    def delete_training_configuration(self, parameter_path, parameter_dict):
        with open(parameter_path) as file:
            trainer_configuration = yaml.safe_load(file)
            del trainer_configuration[parameter_dict]
        with open(parameter_path, 'w') as file:
            yaml.safe_dump(trainer_configuration, file, indent=4)

    def save_training_parameters(self, parameter_path, parameter_dict):
        with open(parameter_path) as file:
            trainer_configuration = yaml.safe_load(file)
            trainer_configuration[parameter_dict] = self.trainer_configuration
        with open(parameter_path, 'w') as file:
            yaml.safe_dump(trainer_configuration, file, indent=4)
        return True

    def get_agent_configuration(self):
        self.agent_configuration = Agent.get_config()
        return self.agent_configuration

    def boost_exploration(self):
        if not self.agent.boost_exploration():
            self.exploration_algorithm.boost_exploration()

    def get_exploration_configuration(self):
        self.exploration_configuration = ExplorationAlgorithm.get_config()
        return self.exploration_configuration

    def change_interface(self, interface):
        global AgentInterface
        if interface == "MLAgentsV17":
            from ..interfaces.mlagents_v17 import MlAgentsV17Interface as AgentInterface
            self.connect = self.connect_to_unity_environment
        elif interface == "OpenAIGym":
            from ..interfaces.openaigym import OpenAIGymInterface as AgentInterface
            self.connect = self.connect_to_gym_environment
        else:
            raise ValueError("An interface for {} is not (yet) supported by this trainer. "
                             "You can implement an interface yourself by utilizing the interface blueprint class "
                             "in the respective folder. "
                             "After that add the respective if condition here.".format(interface))

    def change_training_algorithm(self, training_algorithm):
        global Agent
        if training_algorithm == "DQN":
            from ..training_algorithms.DQN import DQNAgent as Agent
        elif training_algorithm == "A2C":
            from ..training_algorithms.A2C import A2CAgent as Agent
        elif training_algorithm == "DDPG":
            from ..training_algorithms.DDPG import DDPGAgent as Agent
        elif training_algorithm == "TD3":
            from ..training_algorithms.TD3 import TD3Agent as Agent
        elif training_algorithm == "SAC":
            from ..training_algorithms.SAC import SACAgent as Agent
        elif training_algorithm == "PPO":
            from ..training_algorithms.PPO import PPOAgent as Agent
        else:
            raise ValueError("There is no {} training algorithm.".format(training_algorithm))

    def change_exploration_algorithm(self, exploration_algorithm):
        global ExplorationAlgorithm
        if exploration_algorithm == "EpsilonGreedy":
            from ..exploration_algorithms.epsilon_greedy import EpsilonGreedy as ExplorationAlgorithm
        elif exploration_algorithm == "None":
            from ..exploration_algorithms.exploration_algorithm_blueprint import ExplorationAlgorithm
        elif exploration_algorithm == "PseudoCount":
            from ..exploration_algorithms.pseudo_counts import PseudoCount as ExplorationAlgorithm
        elif exploration_algorithm == "ICM":
            from ..exploration_algorithms.intrinsic_curiosity_module import IntrinsicCuriosityModule as ExplorationAlgorithm
        else:
            raise ValueError("There is no {} exploration algorithm.".format(exploration_algorithm))

    def change_curriculum_strategy(self, curriculum_strategy):
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

    def validate_trainer_configuration(self):
        return Agent.validate_config(self.trainer_configuration,
                                     self.agent_configuration,
                                     self.exploration_configuration)

    def validate_action_space(self):
        return Agent.validate_action_space(self.agent_configuration, self.environment_configuration)

    def instantiate_agent(self, mode, model_path=None):
        self.agent = Agent(mode,
                           learning_parameters=self.trainer_configuration,
                           environment_configuration=self.environment_configuration,
                           network_parameters=self.trainer_configuration.get("NetworkParameters"),
                           model_path=model_path)

    def instantiate_logger(self, tensorboard=True):
        self.logging_name = \
            datetime.strftime(datetime.now(), '%y%m%d_%H%M%S_') + self.trainer_configuration['TrainingID']
        self.logger = Logger(os.path.join("training/summaries", self.logging_name),
                             agent_num=self.environment_configuration["AgentNumber"],
                             mode=self.agent_configuration["ReplayBuffer"],
                             tensorboard=tensorboard)
        if tensorboard:
            with open(os.path.join("./training/summaries", self.logging_name, "trainer_config.yaml"), 'w') as file:
                _ = yaml.dump(self.trainer_configuration, file)
                _ = yaml.dump(self.environment_configuration, file)
                _ = yaml.dump(self.exploration_configuration, file)
        return self.logger

    def get_elapsed_training_time(self):
        return self.logger.get_elapsed_time()

    def instantiate_replay_buffer(self):
        if self.agent_configuration["ReplayBuffer"] == "memory":
            self.replay_buffer = ReplayBuffer(capacity=self.trainer_configuration["ReplayCapacity"],
                                              agent_num=self.environment_configuration["AgentNumber"],
                                              n_steps=self.trainer_configuration.get("NSteps"),
                                              gamma=self.trainer_configuration["Gamma"],
                                              mode=self.agent_configuration["ReplayBuffer"],
                                              prioritized=self.trainer_configuration["PrioritizedReplay"])
        else:
            self.replay_buffer = ReplayBuffer(capacity=10000,
                                              agent_num=self.environment_configuration["AgentNumber"],
                                              n_steps=1,
                                              gamma=self.trainer_configuration["Gamma"],
                                              mode=self.agent_configuration["ReplayBuffer"])
        return self.replay_buffer

    def instantiate_exploration_algorithm(self):
        self.exploration_algorithm = ExplorationAlgorithm(self.environment_configuration["ActionShape"],
                                                          self.environment_configuration["ObservationShapes"],
                                                          self.environment_configuration["ActionType"],
                                                          self.trainer_configuration["ExplorationParameters"])
        return self.exploration_algorithm

    def instantiate_curriculum_strategy(self):
        self.curriculum_strategy = CurriculumStrategy()

    def save_agent_models(self, training_step):
        self.agent.save_checkpoint(os.path.join("training/summaries", self.logging_name),
                                   self.logger.best_running_average_reward,
                                   training_step, save_all_models=self.save_all_models)

    def training_loop_generator(self):
        training_step = 0
        training_metrics = {}

        # Check current Task properties
        self.curriculum_strategy.update_task_properties()

        AgentInterface.reset(self.env)
        # 1. Obtain first environment observation
        decision_steps, terminal_steps = AgentInterface.get_steps(self.env, self.behavior_name)
        while True:
            # 2. Choose an action either by exploration algorithm or agent
            actions = self.exploration_algorithm.act(decision_steps)
            if actions is None:
                actions = self.agent.act(decision_steps.obs, mode="training")

            # 3. Add new Observations to Replay Buffer
            self.replay_buffer.add_new_steps(terminal_steps.obs, terminal_steps.reward,
                                             terminal_steps.agent_id, add_to_done=self.test_flag,
                                             step_type="terminal")
            self.replay_buffer.add_new_steps(decision_steps.obs, decision_steps.reward,
                                             decision_steps.agent_id, add_to_done=self.test_flag,
                                             actions=actions, step_type="decision")

            # 4. Track Rewards
            self.logger.track_episode(terminal_steps.reward, terminal_steps.agent_id, add_to_done=self.test_flag,
                                      step_type="terminal")
            self.logger.track_episode(decision_steps.reward, decision_steps.agent_id, add_to_done=self.test_flag,
                                      step_type="decision")

            # Train the agent if enough samples have been collected
            if self.replay_buffer.check_training_condition(self.trainer_configuration):
                # Some exploration algorithms modify samples from the replay buffer
                if self.exploration_algorithm.calculate_intrinsic_reward(self.replay_buffer):
                    self.replay_buffer.new_unmodified_samples = 0
                # Gives the ability to work with a prioritized replay buffer
                self.replay_buffer.update_parameters(training_step)
                # Obtain replay buffer or trajectory samples
                samples, indices = self.replay_buffer.sample(self.trainer_configuration.get("BatchSize"))
                samples = self.exploration_algorithm.get_intrinsic_reward(samples)
                # Train the agent
                training_metrics, sample_errors, training_step = self.agent.learn(samples)
                self.replay_buffer.update_priorities(indices, sample_errors)
                self.exploration_algorithm.learning_step()
                # Log training results to Tensorboard and console
                self.logger.log_dict(training_metrics, training_step)
                self.logger.log_dict(self.exploration_algorithm.get_logs(), training_step)

                if self.test_frequency and training_step % self.test_frequency == 0:
                    self.test_flag = True

            mean_episode_length, mean_episode_reward, episodes, new_episodes = \
                self.logger.get_episode_stats(track_stats=True)

            if mean_episode_length or mean_episode_reward:
                self.curriculum_strategy.check_task_level_change_condition(self.logger.episode_reward_memory,
                                                                           self.logger.episodes_played_memory)
                task_level, average_episodes, average_reward = self.curriculum_strategy.get_logs()
                # Log episode results to Tensorboard
                self.logger.log_scalar("Performance/TrainingEpisodeLength", mean_episode_length, episodes)
                self.logger.log_scalar("Performance/TrainingReward", mean_episode_reward, episodes)
                self.logger.log_scalar("Performance/Tasklevel", task_level, episodes)

                yield mean_episode_length, mean_episode_reward, episodes, training_step, \
                    training_metrics.get("Losses/Loss"), average_episodes, task_level, average_reward

                if not self.exploration_algorithm.prevent_checkpoint():
                    if self.logger.check_early_stopping_condition():
                        self.save_agent_models(training_step)
                        if self.remove_old_checkpoints:
                            self.logger.remove_old_checkpoints()
                        print("Early stopping!")
                        break

                    if self.logger.check_checkpoint_condition():
                        self.save_agent_models(training_step)
                        if self.remove_old_checkpoints:
                            self.logger.remove_old_checkpoints()
                        print("New checkpoint saved!")

                if episodes >= self.trainer_configuration.get("Episodes"):
                    if self.remove_old_checkpoints:
                        self.logger.remove_old_checkpoints()
                    break

            # Check for environment reset condition
            if self.replay_buffer.check_reset_condition():
                if self.test_flag:
                    self.test_flag = False
                    self.play_test_episodes()

                # 6a. Reset the environment
                AgentInterface.reset(self.env)
                self.logger.done_indices.clear()
                self.replay_buffer.done_indices.clear()

            else:
                # 6b. Take a step in the environment
                AgentInterface.step_action(self.env, self.behavior_name, actions)
            # 7. Obtain observation from the environment
            decision_steps, terminal_steps = AgentInterface.get_steps(self.env, self.behavior_name)

    def play_test_episodes(self):
        AgentInterface.reset(self.env)
        self.logger.clear_buffer()
        self.replay_buffer.done_indices.clear()
        decision_steps, terminal_steps = AgentInterface.get_steps(self.env, self.behavior_name)
        test_episodes = 0

        while True:
            # 2. Choose an action either by exploration algorithm or agent
            actions = self.agent.act(decision_steps.obs, mode="testing")

            # 3. Track Rewards
            self.logger.track_episode(terminal_steps.reward, terminal_steps.agent_id,
                                      step_type="terminal")
            self.logger.track_episode(decision_steps.reward, decision_steps.agent_id,
                                      step_type="decision")

            mean_episode_length, mean_episode_reward, episodes, new_episodes = \
                self.logger.get_episode_stats(track_stats=False)
            if mean_episode_length:
                self.played_test_episodes += new_episodes
                self.logger.log_scalar("Performance/TestEpisodeLength", mean_episode_length, self.played_test_episodes)
                self.logger.log_scalar("Performance/TestReward", mean_episode_reward, self.played_test_episodes)
                test_episodes += new_episodes
            if test_episodes >= self.test_episodes:
                self.logger.clear_buffer()
                return

            if self.replay_buffer.check_reset_condition():
                # 6a. Reset the environment
                AgentInterface.reset(self.env)
            else:
                AgentInterface.step_action(self.env, self.behavior_name, actions)
            # 7. Obtain observation from the environment
            decision_steps, terminal_steps = AgentInterface.get_steps(self.env, self.behavior_name)

    def testing_loop_generator(self):
        AgentInterface.reset(self.env)
        # 1. Obtain first environment observation
        decision_steps, terminal_steps = AgentInterface.get_steps(self.env, self.behavior_name)

        while True:
            if self.environment_selection:
                self.env.render()
            # 2. Choose an action either by exploration algorithm or agent
            actions = self.agent.act(decision_steps.obs, mode="testing")

            # 4. Add new Observations to Replay Buffer
            self.replay_buffer.add_new_steps(terminal_steps.obs, terminal_steps.reward,
                                             terminal_steps.agent_id,
                                             step_type="terminal")
            self.replay_buffer.add_new_steps(decision_steps.obs, decision_steps.reward,
                                             decision_steps.agent_id,
                                             actions=actions, step_type="decision")

            # 5. Track Rewards
            self.logger.track_episode(terminal_steps.reward, terminal_steps.agent_id,
                                      step_type="terminal")
            self.logger.track_episode(decision_steps.reward, decision_steps.agent_id,
                                      step_type="decision")

            mean_episode_length, mean_episode_reward, episodes, new_episodes = \
                self.logger.get_episode_stats(track_stats=True)
            if mean_episode_length:
                yield mean_episode_length, mean_episode_reward, episodes, 0, 0, 0, 0, 0
                if episodes >= self.trainer_configuration.get("Episodes"):
                    break

            # Check for environment reset condition
            if self.replay_buffer.check_reset_condition():
                # 6. Reset the environment
                AgentInterface.reset(self.env)
                self.logger.done_indices.clear()
                self.replay_buffer.done_indices.clear()
            else:
                # 6. Take a step in the environment
                AgentInterface.step_action(self.env, self.behavior_name, actions)
            # 7. Obtain observation from the environment
            decision_steps, terminal_steps = AgentInterface.get_steps(self.env, self.behavior_name)


def main():
    pass


if __name__ == '__main__':
    main()
