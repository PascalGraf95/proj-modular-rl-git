#!/usr/bin/env python
import logging

from modules.misc.replay_buffer import FIFOBuffer, PrioritizedBuffer
from modules.misc.logger import GlobalLogger
from modules.misc.utility import getsize, get_exploration_policies

import os
import numpy as np
from datetime import datetime
import time

import yaml
import ray

from pynput.keyboard import Key, Listener


class Trainer:
    """ Modular Reinforcement Learning© Trainer-class
    This base class contains all the relevant functions to acquire information from the environment and other modules
    so that agents can be constructed and algorithms can be adjusted accordingly.
    Furthermore, this class controls the training and testing loop and validates the given parameters.
    """

    def __init__(self):
        # - Trainer Configuration -
        # The trainer configuration file contains all training parameters relevant for the chosen algorithm under the
        # selected key. Furthermore, the abbreviation of the chosen algorithm is stored as it might affect the
        # initiation of other modules.
        self.trainer_configuration = None
        self.training_algorithm = None

        # - Global Tensorboard Logger -
        # The global logger is used to store the training results like rewards and losses in a Tensorboard which can be
        # viewed by typing "tensorboard --logdir ." in a console from the "training/summaries" directory of this
        # project.
        self.global_logger = None
        self.logging_name = None
        self.logging_frequency = 100

        # - Environment -
        # The environment has its own set of parameters like the observation and action shapes as well as the number
        # of agents present. This configuration will be read from the environment right after establishing the
        # connection.
        self.environment_configuration = None

        # - Agent -
        # The agent contains the intelligence of the whole algorithm. It's structure depends on the chosen RL
        # algorithm as well as the trainer configuration file. It consists of one global learner and a list of one or
        # more parallel actors.
        self.actors = None
        self.learner = None
        self.agent_configuration = None

        # - Exploration Algorithm -
        # The exloration algorithm has its own set of parameters defining the extent of exploration as well as a
        # possible decay over time.
        self.exploration_configuration = None
        self.exploration_algorithm = None

        # - Curriculum Learning Strategy -
        # The curriculum strategy determines when an agent has solved the current task level and is ready to proceed
        # to a higher difficulty.
        self.global_curriculum_strategy = None

        # - Preprocessing Algorithm -
        # The preprocessing algorithm usually decreases the input size by extracting relevant information. The agent
        # will then only see this modified information when training and acting.
        self.preprocessing_algorithm = None

        # - ReplayBuffer -
        # All algorithms currently supported by this framework are so called "off-policy algorithms". This means that
        # experiences collected in the past can be used to train the model. In the global buffer experiences from all
        # actors are collected. The trainer then samples a random batch from these experiences and trains the learner
        # with them. The global buffer has a maximum capacity after which old samples will be deleted.
        self.global_buffer = None

        # - Misc -
        self.save_all_models = False
        self.remove_old_checkpoints = False
        self.interface = None

    # region Algorithm Choice
    def select_training_algorithm(self, training_algorithm):
        """ Imports the agent (actor, learner) blueprint according to the algorithm choice."""
        global Actor, Learner
        self.training_algorithm = training_algorithm
        if training_algorithm == "DQN":
            from .training_algorithms.DQN import DQNActor as Actor
            from .training_algorithms.DQN import DQNLearner as Learner
        elif training_algorithm == "DDPG":
            from .training_algorithms.DDPG import DDPGAgent as Agent
        elif training_algorithm == "TD3":
            from .training_algorithms.TD3 import TD3Agent as Agent
        elif training_algorithm == "SAC":
            from .training_algorithms.SAC import SACActor as Actor
            from .training_algorithms.SAC import SACLearner as Learner
        elif training_algorithm == "CQL":
            from .training_algorithms.CQL import CQLActor as Actor
            from .training_algorithms.CQL import CQLLearner as Learner
        else:
            raise ValueError("There is no {} training algorithm.".format(training_algorithm))

    def select_curriculum_strategy(self, curriculum_strategy):
        """Imports the curriculum strategy according to the algorithm choice."""
        global CurriculumStrategy

        if curriculum_strategy == "None" or not curriculum_strategy:
            from .curriculum_strategies.curriculum_strategy_blueprint import NoCurriculumStrategy as CurriculumStrategy
        elif curriculum_strategy == "LinearCurriculum":
            from .curriculum_strategies.linear_curriculum import LinearCurriculum as CurriculumStrategy
        else:
            raise ValueError("There is no {} curriculum strategy.".format(curriculum_strategy))
        """
        Currently Disabled:
        elif curriculum_strategy == "RememberingCurriculum":
            from ..curriculum_strategies.remembering_curriculum import RememberingCurriculum as CurriculumStrategy
        elif curriculum_strategy == "CrossFadeCurriculum":
            from ..curriculum_strategies.cross_fade_curriculum import CrossFadeCurriculum as CurriculumStrategy
        """
    # endregion

    # region Validation
    def validate_action_space(self):
        """Validate that the action spaces of the selected algorithm and the connected environment are compatible.

        :return: None
        """
        return Learner.validate_action_space(self.agent_configuration, self.environment_configuration)
    # endregion

    # region Instantiation
    def async_instantiate_agent(self, mode: str, preprocessing_algorithm: str, exploration_algorithm: str,
                                environment_path: str = None, model_path: str = None,
                                preprocessing_path: str = None, demonstration_path: str = None, clone_path: str = None):
        """ Instantiate the agent consisting of learner and actor(s) and their respective submodules in an asynchronous
        fashion utilizing the ray library."""
        # Initialize ray for parallel multiprocessing.
        ray.init()
        # Alternatively, use the following code line to enable debugging (with ray >= 2.0.X)
        # ray.init(logging_level=logging.INFO, local_mode=True)

        # If the connection is established directly with the Unity Editor or if we are in testing mode, override
        # the number of actors with 1.
        if not environment_path or mode == "testing" or mode == "fastTesting" or environment_path == "Unity":
            actor_num = 1
        else:
            actor_num = self.trainer_configuration.get("ActorNum")

        # If there is only one actor in training mode the degree of exploration should start at maximum. Otherwise,
        # each actor will be instantiated with a different degree of exploration. This only has an effect if the
        # exploration algorithm is not None. When testing mode is selected, thus the number of actors is 1, linspace
        # returns 0. If the Meta-Controller is used, a family of exploration policies is created, where the controller
        # later on can dynamically choose from.
        intrinsic_exploration_algorithms = ["ENM"]  # Will contain NGU, ECR, NGU-r, RND-alter with future updates...

        # Calculate exploration policy values based on agent57's concept
        # The exploration is now a list that contains a dictionary for each actor.
        # Each dictionary contains a value for beta and gamma (utilized for intrinsic motivation)
        # as well as a scaling values (utilized for epsilon greedy).
        exploration_degree = get_exploration_policies(num_policies=actor_num,
                                                      mode=mode,
                                                      beta_max=self.trainer_configuration[
                                                          "ExplorationParameters"].get(
                                                          "MaxIntRewardScaling"))

        # Pass the exploration degree to the trainer configuration
        self.trainer_configuration["ExplorationParameters"]["ExplorationDegree"] = exploration_degree

        # Extension of agent inputs to accept values in addition to the environment observations (exploration policy
        # index, ext. reward, int. reward) is only compatible with usage of intrinsic exploration algorithms.
        if exploration_algorithm not in intrinsic_exploration_algorithms and mode == "training":
            self.trainer_configuration["IntrinsicExploration"] = False
            self.trainer_configuration["PolicyFeedback"] = False
            self.trainer_configuration["RewardFeedback"] = False
            print("\n\nExploration policy feedback and reward feedback automatically disabled as no intrinsic "
                  "exploration algorithm is in use.\n\n")
        else:
            self.trainer_configuration["IntrinsicExploration"] = True

        # - Actor Instantiation -
        # Create the desired number of actors using the ray "remote"-function. Each of them will construct their own
        # environment, exploration algorithm and preprocessing algorithm instances. The actors are distributed along
        # the cpu cores and threads, which means their number in the trainer configuration should not extend the maximum
        # number of cpu threads - 2.
        self.actors = [Actor.remote(idx, 5004 + idx, mode,
                                    self.interface,
                                    preprocessing_algorithm,
                                    preprocessing_path,
                                    exploration_algorithm,
                                    environment_path,
                                    demonstration_path,
                                    '/cpu:0') for idx in range(actor_num)]

        # Instantiate one environment for each actor and connect them to one another.
        if self.interface == "MLAgentsV18":
            [actor.connect_to_unity_environment.remote() for actor in self.actors]
        else:
            [actor.connect_to_gym_environment.remote() for actor in self.actors]

        # For each actor instantiate the necessary modules
        for i, actor in enumerate(self.actors):
            actor.instantiate_modules.remote(self.trainer_configuration, exploration_degree)
            # In case of Unity Environments set the rendering and simulation parameters.
            if self.interface == "MLAgentsV18":
                if mode == "training":
                    actor.set_unity_parameters.remote(time_scale=1000,
                                                      width=10, height=10,
                                                      quality_level=1,
                                                      target_frame_rate=-1)
                elif mode == "fastTesting":
                    actor.set_unity_parameters.remote(time_scale=1000, width=500, height=500)
                else:
                    actor.set_unity_parameters.remote(time_scale=1, width=500, height=500)

        # Get the environment and exploration configuration from the first actor.
        # NOTE: ray remote-functions return IDs only. If you want the actual returned value of the function you need to
        # call ray.get() on the ID.
        environment_configuration = self.actors[0].get_environment_configuration.remote()
        self.environment_configuration = ray.get(environment_configuration)
        exploration_configuration = self.actors[0].get_exploration_configuration.remote()
        self.exploration_configuration = ray.get(exploration_configuration)

        # - Learner Instantiation -
        # Create one learner capable of learning according to the selected algorithm utilizing the buffered actor
        # experiences. If a model path is given, the respective networks will continue to learn from this checkpoint.
        # If a clone path is given, it will be used as starting point for self-play.
        self.learner = Learner.remote(mode, self.trainer_configuration,
                                      self.environment_configuration,
                                      model_path,
                                      clone_path)

        # Initialize the actor network for each actor
        network_ready = [actor.build_network.remote(self.trainer_configuration.get("NetworkParameters"),
                                                    self.environment_configuration)
                         for idx, actor in enumerate(self.actors)]
        # Wait for the networks to be built
        ray.wait(network_ready)

        # - Global Buffer -
        # The global buffer stores the experiences from each actor which will be sampled during the training loop in
        # order to train the Learner networks. If the prioritized replay buffer is enabled, samples will not be chosen
        # with uniform probability but considering their priority, i.e. their "unpredictability".
        if self.trainer_configuration["PrioritizedReplay"]:
            self.global_buffer = PrioritizedBuffer.remote(capacity=self.trainer_configuration["ReplayCapacity"],
                                                          priority_alpha=self.trainer_configuration["PriorityAlpha"])
        else:
            self.global_buffer = FIFOBuffer.remote(capacity=self.trainer_configuration["ReplayCapacity"])
        if mode == "training":
            # Initialize the global curriculum strategy, but only in training mode
            self.global_curriculum_strategy = CurriculumStrategy.remote()

        # - Global Logger -
        # The global logger writes the current training stats to a tensorboard event file which can be viewed in the
        # browser. This will only be done, if the agent is in training mode.
        self.logging_name = \
            datetime.strftime(datetime.now(), '%y%m%d_%H%M%S_') + self.trainer_configuration['TrainingID']
        self.global_logger = GlobalLogger.remote(
            os.path.join("training/summaries", self.logging_name),
            actor_num=self.trainer_configuration["ActorNum"],
            tensorboard=(mode == 'training'),
            periodic_model_saving=(self.training_algorithm == 'CQL' or
                                   self.environment_configuration.get("BehaviorCloneName")))
        # If training mode is enabled all configs are stored into a yaml file in the summaries folder
        if mode == 'training':
            if not os.path.isdir(os.path.join("./training/summaries", self.logging_name)):
                os.makedirs(os.path.join("./training/summaries", self.logging_name))
            with open(os.path.join("./training/summaries", self.logging_name, "training_parameters.yaml"), 'w') as file:
                _ = yaml.dump(self.trainer_configuration, file)
                _ = yaml.dump(self.environment_configuration, file)
                _ = yaml.dump(self.exploration_configuration, file)
    # endregion

    # region Misc
    def async_save_agent_models(self, training_step):
        """"""
        checkpoint_condition = self.global_logger.check_checkpoint_condition.remote(training_step)
        self.learner.save_checkpoint.remote(os.path.join("training/summaries", self.logging_name),
                                            self.global_logger.get_best_running_average.remote(),
                                            training_step, self.save_all_models, checkpoint_condition)
        if self.remove_old_checkpoints:
            self.global_logger.remove_old_checkpoints.remote()

    def parse_training_parameters(self, parameter_path, parameter_dict=None):
        """"""
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
        """"""
        # region --- Initialization ---
        total_episodes = 0
        training_step = 0
        # Before starting the acting and training loop all actors need a copy of the latest network weights.
        [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(True), total_episodes)
         for actor in self.actors]
        # endregion

        while True:
            # region --- Acting ---
            # Receiving the latest state from its environment each actor chooses an action according to its policy
            # and/or its exploration algorithm. Furthermore, the reward and the info about the environment state (done)
            # is processed and stored in a local replay buffer for each actor.
            actors_ready = [actor.play_one_step.remote(training_step) for actor in self.actors]
            # endregion

            # region --- Global Buffer and Logger ---
            # If an actor has collected enough samples, copy the samples from its local buffer to the global buffer.
            # In case of Prioritized Experience Replay let the actor calculate an initial priority first.
            sample_error_list = []
            for idx, actor in enumerate(self.actors):
                # Gather samples from the local buffer of an actor.
                samples, indices = actor.get_new_samples.remote()
                # Calculate an initial priority based on the temporal difference error of the critic network.
                if self.trainer_configuration["PrioritizedReplay"]:
                    sample_errors = actor.get_sample_errors.remote(samples)
                    sample_error_list.append(sample_errors)
                    # Append the received samples to the global buffer.
                    self.global_buffer.append_list.remote(samples, sample_errors)
                else:
                    # Append the received samples to the global buffer.
                    self.global_buffer.append_list.remote(samples)
                # Request the latest episode stats from the actor containing episode rewards and lengths and append them
                # to the global logger which will write them to the tensorboard.
                self.global_logger.append.remote(*actor.get_new_stats.remote(), actor_idx=idx)
            # endregion

            # region --- Training Process ---
            # This code section is responsible for the actual training process of the involved neural networks.
            # Sample a new batch of transitions from the global replay buffer. Their indices matter in case of a
            # Prioritized Replay Buffer (PER) because their priorities need to be updated afterwards.
            samples, indices = self.global_buffer.sample.remote(self.trainer_configuration,
                                                                self.trainer_configuration.get("BatchSize"))

            # Some exploration algorithms assign an intrinsic reward on top of the extrinsic reward to each sample.
            # This reward is usually based upon the novelty of a visited state and aims to promote exploring the
            # environment.
            samples = self.actors[0].get_intrinsic_rewards.remote(samples)

            # Train the learner networks with the batch and observe the resulting metrics.
            training_metrics, sample_errors, training_step = self.learner.learn.remote(samples)

            # In case of a PER update the buffer priorities with the temporal difference errors
            self.global_buffer.update.remote(indices, sample_errors)
            # Train the actor's exploration algorithm with the same batch
            [actor.exploration_learning_step.remote(samples) for actor in self.actors]

            # If the training algorithm is constructed to act and learn by utilizing a recurrent neural network a
            # sequence length is chosen in the training parameters. However, as training progresses, the episode length
            # might change for some environments allowing for only shorter or even longer training sequences. If the
            # respective option is enabled, this function automatically adapts the sequence length.
            self.async_adapt_sequence_length(training_step)
            # endregion

            # region --- Network Update and Checkpoint Saving ---
            # Check if the actor networks request to be updated with the latest network weights from the learner.
            [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(
                actor.is_network_update_requested.remote(training_step)), self.global_logger.get_total_episodes.remote())
                for actor in self.actors]

            # Check if a new best reward has been achieved, if so save the models
            self.async_save_agent_models(training_step)
            # endregion
            # region --- Tensorboard Logging ---
            # Every logging_frequency steps (usually set to 100 so the event file doesn't get to big for long trainings)
            # log the training metrics to the tensorboard.
            self.global_logger.log_dict.remote(training_metrics, training_step, self.logging_frequency)
            self.global_logger.log_dict.remote({"Losses/BufferLength": ray.get(self.global_buffer.__len__.remote())},
                                               training_step, 10)
            # Some exploration algorithms also return metrics that represent their training or decay state. These shall
            # be logged in the same interval.
            for idx, actor in enumerate(self.actors):
                self.global_logger.log_dict.remote(actor.get_exploration_logs.remote(), training_step,
                                                   self.logging_frequency)
            # endregion

            # region --- Waiting ---
            # When asynchronous ray processes don't finish in time they are added to a queue while the next loop starts.
            # This leads to a memory leak filling up the RAM and after that even taking up hard drive storage. To
            # prevent this behavior we explicitly wait for the most resource and time-consuming processes at the end of
            # each iteration.
            ray.wait(actors_ready)
            ray.wait([training_metrics])
            ray.wait(sample_error_list)
            ray.wait([sample_errors])
            # endregion

            # TODO: Add an option to affect the training process with keyboard events.
            #       E.g. Skip Level, Boost Exploration, Render Test Episode

    def on_press(self, key):
        try:
            x = key.char
        except AttributeError:
            return

        if key.char == "t":
            print("Testing Mode")
            if self.interface == "MLAgentsV18":
                self.actors[0].set_unity_parameters.remote(time_scale=1,
                                                           width=500, height=500,
                                                           quality_level=5,
                                                           target_frame_rate=60,
                                                           capture_frame_rate=60)
        elif key.char == "z":
            print("Training Mode")
            if self.interface == "MLAgentsV18":
                self.actors[0].set_unity_parameters.remote(time_scale=1000,
                                                           width=10, height=10,
                                                           quality_level=1,
                                                           target_frame_rate=-1,
                                                           capture_frame_rate=60)

    def async_adapt_sequence_length(self, training_step):
        """"""
        # Only applies if the network is recurrent and the respective flag is set to true
        if self.trainer_configuration["Recurrent"] and self.trainer_configuration["AdaptiveSequenceLength"]:
            # Check if a new sequence length is recommended.
            new_sequence_length, new_burn_in = self.global_logger.get_new_sequence_length.remote(
                self.trainer_configuration["SequenceLength"], self.trainer_configuration["BurnIn"],
                training_step)
            new_sequence_length = ray.get(new_sequence_length)
            new_burn_in = ray.get(new_burn_in)
            # If so, update the trainer configuration, the actors and the learner. Also, reset the global buffer.
            if new_sequence_length:
                self.trainer_configuration["SequenceLength"] = new_sequence_length
                self.trainer_configuration["BurnIn"] = new_burn_in
                [actor.update_sequence_length.remote(self.trainer_configuration) for actor in self.actors]
                self.learner.update_sequence_length.remote(self.trainer_configuration)
                self.global_buffer.reset.remote()

    def async_training_loop_curriculum(self):
        """"""
        # region --- Initialization ---
        total_episodes = 0
        # If there is a learning curriculum, the properties of the initial task have to be acquired first.
        # This is done by connecting to and reading from a custom side channel. The information about the task contain
        # the total number of tasks, the current task level, the number of episode to average the reward over and the
        # transition value to proceed to the next task.
        unity_responded, task_properties = self.actors[0].get_task_properties.remote(False)
        # If Unity responded to the request, i.e. if there is a side channel and a curriculum, update the global
        # curriculum strategy instance.
        self.global_curriculum_strategy.update_task_properties.remote(unity_responded, task_properties)

        # Before starting the acting and training loop all actors need a copy of the latest network weights.
        [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(True), total_episodes)
         for actor in self.actors]
        # endregion

        while True:
            # region --- Acting ---
            # Receiving the latest state from its environment each actor chooses an action according to its policy
            # and/or its exploration algorithm. Furthermore, the reward and the info about the environment state (done)
            # is processed and stored in a local replay buffer for each actor.
            actors_ready = [actor.play_one_step.remote() for actor in self.actors]
            # endregion

            # region --- Curriculum Update ---
            # This part of the code checks if all environments are up-to-date in terms of the current task level.
            # First it checks if Unity responded with the latest task information. If not then the first actor requests
            # the latest information from its environment. The Unity responded global flag is reset upon level change.
            unity_responded_global = self.global_curriculum_strategy.has_unity_responded.remote()
            unity_responded, task_properties = self.actors[0].get_task_properties.remote(unity_responded_global)
            # If Unity responded to the request update the global curriculum strategy instance.
            self.global_curriculum_strategy.update_task_properties.remote(unity_responded, task_properties)
            # If
            [actor.send_target_task_level.remote(unity_responded_global) for actor in self.actors]
            # endregion

            # region --- Global Buffer and Logger ---
            # If an actor has collected enough samples, copy the samples from its local buffer to the global buffer.
            # In case of Prioritized Experience Replay let the actor calculate an initial priority first.
            sample_error_list = []
            for idx, actor in enumerate(self.actors):
                # Gather samples from the local buffer of an actor.
                samples, indices = actor.get_new_samples.remote()
                # Calculate an initial priority based on the temporal difference error of the critic network.
                if self.trainer_configuration["PrioritizedReplay"]:
                    sample_errors = actor.get_sample_errors.remote(samples)
                    sample_error_list.append(sample_errors)
                    # Append the received samples to the global buffer.
                    self.global_buffer.append_list.remote(samples, sample_errors)
                else:
                    # Append the received samples to the global buffer.
                    self.global_buffer.append_list.remote(samples)
                # Request the latest episode stats from the actor containing episode rewards and lengths and append them
                # to the global logger which will write them to the tensorboard.
                self.global_logger.append.remote(*actor.get_new_stats.remote(), actor_idx=idx)
            # endregion

            # region --- Training Process ---
            # This code section is responsible for the actual training process of the involved neural networks.

            # Sample a new batch of transitions from the global replay buffer. Their indices matter in case of a
            # Prioritized Replay Buffer (PER) because their priorities need to be updated afterwards.
            samples, indices = self.global_buffer.sample.remote(self.trainer_configuration,
                                                                self.trainer_configuration.get("BatchSize"))
            # Some exploration algorithms assign an intrinsic reward on top of the extrinsic reward to each sample.
            # This reward is usually based upon the novelty of a visited state and aims to promote exploring the
            # environment.
            samples = self.actors[0].get_intrinsic_rewards.remote(samples)

            # Train the learner networks with the batch and observe the resulting metrics.
            training_metrics, sample_errors, training_step = self.learner.learn.remote(samples)

            # In case of a PER update the buffer priorities with the temporal difference errors
            self.global_buffer.update.remote(indices, sample_errors)
            # Train the actors exploration algorithm with the same batch
            [actor.exploration_learning_step.remote(samples) for actor in self.actors]
            # endregion

            # region --- Network Update and Checkpoint Saving ---
            # Check if the actor networks request to be updated with the latest network weights from the learner.
            [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(
                actor.is_network_update_requested.remote()), total_episodes)
                for actor in self.actors]

            # Check if a new best reward has been achieved, if so save the models
            self.async_save_agent_models(training_step)
            # endregion

            # Check if the reward threshold for a level change has been reached
            _, max_reward, total_episodes = self.global_logger.get_current_max_stats.remote(
                self.global_curriculum_strategy.get_average_episodes.remote())
            task_level_condition = \
                self.global_curriculum_strategy.check_task_level_change_condition.remote(max_reward,
                                                                                         total_episodes)
            # If level change condition is met, register the level change
            self.global_logger.register_level_change.remote(task_level_condition)
            target_task_level = self.global_curriculum_strategy.get_new_task_level.remote(task_level_condition)
            for actor in self.actors:
                actor.set_new_target_task_level.remote(target_task_level)

            # Log training results to Tensorboard
            self.global_logger.log_dict.remote(training_metrics, training_step, self.logging_frequency)
            for idx, actor in enumerate(self.actors):
                self.global_logger.log_dict.remote(actor.get_exploration_logs.remote(idx), training_step,
                                                   self.logging_frequency)

            # In case of recurrent training, check if the sequence length should be adapted
            self.async_adapt_sequence_length(training_step)

            # Wait for the actors to finish their environment steps and learner to finish the learning step
            ray.wait(actors_ready)
            ray.wait([training_metrics])
            ray.wait(sample_error_list)

    def async_testing_loop(self):
        """"""
        # Before starting the acting loop all actors need a copy of the latest network weights.
        [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(True), 0) for actor in self.actors]
        rating_period = None
        while True:
            # Receiving the latest state from its environment each actor chooses an action according to its policy.
            actors_ready = [actor.play_one_step.remote(0) for actor in self.actors]            
            for actor in self.actors:
                # update the side channel information. For example this is used to update the game results history for the rating algorithm
                # TODO: Get the id of the agents playing against each other from the tournament scheduler
                actor.get_side_channel_information.remote(self.model_path)
                # get the rating period
                if rating_period is None:
                    rating_period = actor.get_current_rating_period.remote(self.model_path)
                # rating period changed
                if rating_period != actor.get_current_rating_period.remote(self.model_path):
                    # update the rating period
                    rating_period = actor.get_current_rating_period.remote(self.model_path)
                    # update the rating history
                    actor.update_ratings.remote(self.model_path)
                
            # Wait for the actors to finish their environment steps
            ray.wait(actors_ready)
    # endregion
