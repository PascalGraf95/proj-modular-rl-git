#!/usr/bin/env python
import logging

from modules.misc.replay_buffer import FIFOBuffer, PrioritizedBuffer
from modules.misc.logger import GlobalLogger
from modules.misc.utility import getsize, get_exploration_policies
from modules.misc.model_path_handling import create_model_dictionary_from_path

import os
import numpy as np
from datetime import datetime
import time

import yaml
import ray

intrinsic_exploration_algorithms = ["ENM", "RND", "NGU"]  # Will contain NGU, ECR, NGU-r, RND-alter with future updates...


class Trainer:
    """ Modular Reinforcement Learning© Trainer-class
    This base class contains all the relevant functions to acquire information from the environment and other modules
    so that agents can be constructed and algorithms can be adjusted accordingly.
    Furthermore, this class controls the training and testing loop and validates the given parameters.
    """

    def __init__(self):
        # region --- Instance Variables ---
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
        # Different interfaces allow the framework to communicate with different types of environments, e.g. the same
        # functionality can be used when communicating with an OpenAI Gym environment as well as with a Unity
        # Environment communicating via MLAgents
        self.interface = None

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
        # A new set of models will be saved during training as soon as the agent improves in terms of reward. As there
        # may be hundreds of models stored after a long training the following option allows to only keep the latest.
        # However, sometimes it is useful to compare different intermediate models.
        self.remove_old_checkpoints = False

        # - Loading Models -
        # Provided a path with network models, these model dictionaries store a unique key for each model present
        # along with information such as the number of steps it has been trained, the reward it reached and the
        # respective paths. During continued training or testing usually the latest trained model is loaded.
        # In tournament mode all models in the dictionaries will be tested and compared.
        self.model_dictionary = None
        self.clone_model_dictionary = None

        # - Self-Play Tournament -
        # In case of self-play where an agent competes with another agent a tournament can be arranged, where each
        # agent in the model dictionary plays against each agent in the clone model dictionary. The tournament
        # schedule stores all fixtures (pairings) to be played.
        self.tournament_schedule = None
        # The fixture index stores information about which game is currently played from the tournament schedule.
        self.current_tournament_fixture_idx = 0
        # To make up for stochastic policy behaviors each pairing should play multiple games.
        self.games_played_in_fixture = 0
        self.games_per_fixture = None
        # After playing a csv file is created storing each game played along with the scores of the players.
        self.history_path = None

        # endregion

    # region Algorithm Choice
    def select_training_algorithm(self, training_algorithm):
        """ Imports the agent (actor, learner) blueprint according to the algorithm choice."""
        global Actor, Learner
        self.training_algorithm = training_algorithm
        if training_algorithm == "DQN":
            from .training_algorithms.DQN import DQNActor as Actor
            from .training_algorithms.DQN import DQNLearner as Learner
        elif training_algorithm == "SAC":
            from .training_algorithms.SAC import SACActor as Actor
            from .training_algorithms.SAC import SACLearner as Learner
        elif training_algorithm == "CQL":
            from .training_algorithms.CQL import CQLActor as Actor
            from .training_algorithms.CQL import CQLLearner as Learner
        else:
            raise ValueError("There is no {} training algorithm.".format(training_algorithm))

    # endregion

    # region Validation
    def validate_action_space(self):
        """Validate that the action spaces of the selected algorithm and the connected environment are compatible.

        :return: None
        """
        return Learner.validate_action_space(self.agent_configuration, self.environment_configuration)
    # endregion

    # region ---Instantiation ---
    def async_instantiate_agent(self, mode: str, preprocessing_algorithm: str, exploration_algorithm: str,
                                environment_path: str = None, preprocessing_path: str = None,
                                demonstration_path: str = None, args=None):
        """
        Instantiate the agent consisting of learner, actor(s) and their respective submodules in an asynchronous
        fashion utilizing the ray library.
        :param mode: Defines if training, testing or tournament will be performed.
        :param preprocessing_algorithm: Defines if and in which way observations will be preprocessed before passed
        through the actor / learner.
        :param exploration_algorithm: Defines which algorithms are used to explore the state-action-space in the
        environment.
        :param environment_path: Can be either None to connect directly to Unity, a path to an exported Unity
        environment, or the name of an OpenAI Gym environment.
        :param preprocessing_path: Respective path for a preprocessing algorithm.
        :param demonstration_path: Respective path for demonstrations utilized in Offline Reinforcement Learning, i.e.,
        the CQL Algorithm.
        :param args: Arguments from command line only used for logging to yaml
        :return:
        """
        # region - Multiprocessing Initialization and Actor Number Determination
        # Initialize ray for parallel multiprocessing.
        ray.init()
        # Alternatively, use the following code line to enable debugging (with ray >= 2.0.X)
        # ray.init(logging_level=logging.INFO, local_mode=True)

        # If the connection is established directly with the Unity Editor or if we are in testing mode, override
        # the number of actors with 1.
        if not environment_path or mode == "testing" or mode == "fastTesting" or mode == "tournament" or \
                environment_path == "Unity":
            actor_num = 1
        else:
            actor_num = self.trainer_configuration.get("ActorNum")
        # endregion

        # region - Exploration Parameter Determination and Network Feedback -
        # The exploration is a list that contains a dictionary for each actor (if multiple).
        # Each dictionary contains a value for beta and gamma (utilized for intrinsic motivation)
        # as well as a scaling value (utilized for epsilon greedy). If there is only one actor in training mode
        # the scaling of exploration should start at maximum.
        exploration_degree = get_exploration_policies(num_policies=actor_num,
                                                      mode=mode,
                                                      beta_max=self.trainer_configuration[
                                                          "ExplorationParameters"].get(
                                                          "MaxIntRewardScaling"))

        # Pass the exploration degree to the trainer configuration
        self.trainer_configuration["ExplorationParameters"]["ExplorationDegree"] = exploration_degree

        # Network feedbacks are extensions of the agents input in addition to the environment observations.
        # Those are a possible intrinsic reward, the external reward, a policy index and the last taken action.
        # Some of them are only compatible with the usage of intrinsic exploration algorithms and should otherwise
        # be deactivated.
        self.adapt_trainer_configuration(exploration_algorithm, mode)
        # endregion

        # region - Actor Instantiation and Environment Connection -
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
            actor.instantiate_modules.remote(self.trainer_configuration)
            # In case of Unity Environments set the rendering and simulation parameters.
            if self.interface == "MLAgentsV18":
                if mode == "training":
                    actor.set_unity_parameters.remote(time_scale=1000,
                                                      width=10, height=10,
                                                      quality_level=1,
                                                      target_frame_rate=-1)
                elif mode == "fastTesting" or mode == "tournament":
                    actor.set_unity_parameters.remote(time_scale=1000, width=500, height=500)
                else:
                    actor.set_unity_parameters.remote(time_scale=1, width=500, height=500)
        # endregion

        # region - Environment & Exploration Configuration Query -
        # Get the environment and exploration configuration from the first actor.
        # NOTE: ray remote-functions return IDs only. If you want the actual returned value of the function you need to
        # call ray.get() on the ID.
        environment_configuration = self.actors[0].get_environment_configuration.remote()
        self.environment_configuration = ray.get(environment_configuration)
        exploration_configuration = self.actors[0].get_exploration_configuration.remote()
        self.exploration_configuration = ray.get(exploration_configuration)
        # endregion

        # region - Learner Instantiation and Network Construction -
        # Create one learner capable of learning according to the selected algorithm utilizing the buffered actor
        # experiences. If a model path is given, the respective networks will continue to learn from this checkpoint.
        # If a clone path is given, it will be used as starting point for self-play.
        self.learner = Learner.remote(mode, self.trainer_configuration,
                                      self.environment_configuration,
                                      self.model_dictionary,
                                      self.clone_model_dictionary)

        # Initialize the actor network for each actor
        network_ready = [actor.build_network.remote(self.trainer_configuration.get("NetworkParameters"),
                                                    self.environment_configuration)
                         for idx, actor in enumerate(self.actors)]
        # Wait for the networks to be built
        ray.wait(network_ready)
        # endregion

        # region - Global Buffer -
        # The global buffer stores the experiences from each actor which will be sampled during the training loop in
        # order to train the Learner networks. If the prioritized replay buffer is enabled, samples will not be chosen
        # with uniform probability but considering their priority, i.e. their "unpredictability".
        if self.trainer_configuration["PrioritizedReplay"]:
            self.global_buffer = PrioritizedBuffer.remote(capacity=self.trainer_configuration["ReplayCapacity"],
                                                          priority_alpha=self.trainer_configuration["PriorityAlpha"])
        else:
            self.global_buffer = FIFOBuffer.remote(capacity=self.trainer_configuration["ReplayCapacity"])
        # endregion

        # region - Global Logger and Log File -
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
        
        # Add additional environment information from side channel if available.
        environment_info = self.actors[0].get_side_channel_information.remote('environment_info')
        
        # If training mode is enabled all configs are stored into a yaml file in the summaries folder
        if mode == 'training':
            if not os.path.isdir(os.path.join("./training/summaries", self.logging_name)):
                os.makedirs(os.path.join("./training/summaries", self.logging_name))
            with open(os.path.join("./training/summaries", self.logging_name, "training_parameters.yaml"), 'w') as file:
                _ = yaml.dump(args, file)
                _ = yaml.dump(self.trainer_configuration, file)
                _ = yaml.dump(ray.get(environment_info), file)
                _ = yaml.dump(self.environment_configuration, file)
                _ = yaml.dump(self.exploration_configuration, file)
        # endregion
    # endregion

    # region Misc
    def create_model_dictionaries(self, model_path, clone_path):
        self.model_dictionary = create_model_dictionary_from_path(model_path)
        self.clone_model_dictionary = create_model_dictionary_from_path(clone_path)

    def create_tournament_schedule(self, return_match=True):
        self.tournament_schedule = list()
        tournament_tag_set = set()
        for model_key in self.model_dictionary.keys():
            for clone_model_key in self.clone_model_dictionary.keys():
                if model_key != clone_model_key:
                    if model_key + "." + clone_model_key not in tournament_tag_set and \
                            (clone_model_key + "." + model_key not in tournament_tag_set or return_match):
                        tournament_tag_set.add(model_key + "." + clone_model_key)
                        self.tournament_schedule.append([model_key, clone_model_key])
        print("Created Tournament Schedule with {} entries".format(len(self.tournament_schedule)))

    def async_save_agent_models(self, training_step):
        """"""
        checkpoint_condition = self.global_logger.check_checkpoint_condition.remote(training_step)
        self.learner.save_checkpoint.remote(os.path.join("training/summaries", self.logging_name),
                                            self.global_logger.get_best_running_average.remote(),
                                            training_step, checkpoint_condition)
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

    def adapt_trainer_configuration(self, exploration_algorithm, mode):
        """
        Adapts the trainer configuration according to the selected exploration algorithm.
        Deactivate some network feedbacks when no intrinsic exploration algorithm is active.
        """
        if exploration_algorithm not in intrinsic_exploration_algorithms and mode == "training":
            self.trainer_configuration["IntrinsicExploration"] = False
            self.trainer_configuration["PolicyFeedback"] = False
            self.trainer_configuration["RewardFeedback"] = False
            print("\n\nExploration policy feedback and reward feedback automatically disabled as no intrinsic "
                  "exploration algorithm is in use.\n\n")
        else:
            self.trainer_configuration["IntrinsicExploration"] = True

    # endregion

    # region Environment Interaction
    def async_training_loop(self):
        """"""
        # region --- Initialization ---
        training_step = 0
        # Before starting the acting and training loop all actors need a copy of the latest network weights.
        [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(True))
         for actor in self.actors]
        # In case of self-play the clone network needs the latest weights as well.
        [actor.update_clone_network.remote(self.learner.get_clone_network_weights.remote(True))
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
                    # Append the received samples to the global buffer along with the sample errors.
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
                actor.is_network_update_requested.remote()))
                for actor in self.actors]
            # Check if the clone networks request to be updated with the latest network weights from the learner.
            clone_updated = [actor.update_clone_network.remote(self.learner.get_clone_network_weights.remote(
                actor.is_clone_network_update_requested.remote(self.global_logger.get_total_episodes.remote()), True))
                for actor in self.actors]
            # If the clone network has been updated, reset the reward threshold for saving new models.
            self.global_logger.reset_threshold_reward.remote(clone_updated[0])

            # Check if a new best reward has been achieved, if so save the models.
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
            # for idx, actor in enumerate(self.actors):
            [self.global_logger.log_dict.remote(
                actor.get_exploration_logs.remote(), training_step, self.logging_frequency)
                for actor in self.actors]
            # endregion

            # region --- Waiting ---
            # When asynchronous ray processes don't finish in time they are added to a queue while the next loop starts.
            # This leads to a memory leak filling up the RAM and after that even taking up hard drive storage. To
            # prevent this behavior we explicitly wait for the most resource and time-consuming processes at the end of
            # each iteration.
            ray.wait(actors_ready)
            ray.wait([training_metrics])
            ray.wait(sample_error_list)
            # endregion

    def async_minimal_loop(self):
        """"""
        # region --- Initialization ---
        training_step = 0
        # Before starting the acting and training loop all actors need a copy of the latest network weights.
        [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(True))
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
                    # Append the received samples to the global buffer along with the sample errors.
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

            # Train the learner networks with the batch and observe the resulting metrics.
            training_metrics, sample_errors, training_step = self.learner.learn.remote(samples)
            # In case of a PER update the buffer priorities with the temporal difference errors
            self.global_buffer.update.remote(indices, sample_errors)
            # endregion

            # region --- Network Update and Checkpoint Saving ---
            # Check if the actor networks request to be updated with the latest network weights from the learner.
            [actor.update_actor_network.remote(self.learner.get_actor_network_weights.remote(
                actor.is_network_update_requested.remote()))
                for actor in self.actors]
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
                [self.global_logger.log_dict.remote(
                    actor.get_exploration_logs.remote(), training_step, self.logging_frequency)
                    for actor in self.actors]
            # endregion

            # region --- Waiting ---
            # When asynchronous ray processes don't finish in time they are added to a queue while the next loop starts.
            # This leads to a memory leak filling up the RAM and after that even taking up hard drive storage. To
            # prevent this behavior we explicitly wait for the most resource and time-consuming processes at the end of
            # each iteration.
            ray.wait(actors_ready)
            ray.wait([training_metrics])
            ray.wait(sample_error_list)
            # endregion

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

    def async_testing_loop(self):
        """"""
        # Before starting the acting loop all actors need a copy of the latest network weights.
        [actor.update_actor_network.remote(
            self.learner.get_actor_network_weights.remote(True)) for actor in self.actors]
        # In case of self-play the clone network needs the latest weights as well.
        [actor.update_clone_network.remote(self.learner.get_clone_network_weights.remote(True))
         for actor in self.actors]

        while True:
            # Receiving the latest state from its environment each actor chooses an action according to its policy.
            actors_ready = [actor.play_one_step.remote(0) for actor in self.actors]
            # Wait for the actors to finish their environment steps
            ray.wait(actors_ready)

    def async_tournament_loop(self):
        # Get the first fixture keys and load the respective models into the learner
        player_keys = self.tournament_schedule[self.current_tournament_fixture_idx]
        self.learner.load_checkpoint_by_mode.remote(player_keys[0], player_keys[1])

        # Before starting the acting loop all actors need a copy of the latest network weights.
        [actor.update_actor_network.remote(
            self.learner.get_actor_network_weights.remote(True)) for actor in self.actors]
        # In case of self-play the clone network needs the latest weights as well.
        [actor.update_clone_network.remote(self.learner.get_clone_network_weights.remote(True))
         for actor in self.actors]

        while True:
            # Receiving the latest state from its environment each actor chooses an action according to its policy.
            actors_ready = [actor.play_one_step.remote(0) for actor in self.actors]
            new_match_played = [actor.update_history_with_latest_game_results.remote(
                history_path=self.history_path,
                player_keys=player_keys) for actor
                in self.actors]
            # Wait for the actors to finish their environment steps
            ray.wait(actors_ready)
            new_match_played = ray.get(new_match_played)
            if new_match_played[0]:
                self.games_played_in_fixture += 1
                if self.games_played_in_fixture >= self.games_per_fixture:
                    self.current_tournament_fixture_idx += 1
                    print("Playing Game {} of {} from tournament schedule".format(self.current_tournament_fixture_idx+1,
                                                                                  len(self.tournament_schedule)))
                    self.games_played_in_fixture = 0

                    if self.current_tournament_fixture_idx >= len(self.tournament_schedule):
                        break

                    player_keys = self.tournament_schedule[self.current_tournament_fixture_idx]
                    self.learner.load_checkpoint_by_mode.remote(player_keys[0], player_keys[1])

                    [actor.update_actor_network.remote(
                        self.learner.get_actor_network_weights.remote(True)) for actor in self.actors]
                    [actor.update_clone_network.remote(self.learner.get_clone_network_weights.remote(True))
                     for actor in self.actors]

    # endregion
