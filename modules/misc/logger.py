import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from collections import deque
from tensorflow.python.summary.summary_iterator import summary_iterator
from datetime import datetime
import ray


class LocalLogger:
    """
    Local Logger
    This logger is created once for each actor and temporarily keeps track of the episode rewards and lengths.
    The loggers are reset after the stats are read out. Only the total number of episodes persists.
    """
    def __init__(self, agent_num=1):
        # Count of episodes played in total
        self.total_episodes_played = 0

        # Number of agents in the environment + list of rewards over the current episode
        self.agent_num = agent_num
        self.agent_reward_list = [[] for x in range(self.agent_num)]
        # Lists for rewards and episode lengths since last readout
        self.new_episode_rewards = []
        self.new_episode_lengths = []

    def track_episode(self, rewards, agent_ids, step_type='decision'):
        # For each agent in the environment, track the rewards of the current episode
        for idx, agent_id in enumerate(agent_ids):
            self.agent_reward_list[agent_id].append(rewards[idx])
            # If the current state is terminal add the summed rewards of the current agent to the buffer and reset
            if step_type == 'terminal':
                self.new_episode_rewards.append(np.sum(self.agent_reward_list[agent_id]))
                self.new_episode_lengths.append(len(self.agent_reward_list[agent_id]))
                self.agent_reward_list[agent_id].clear()

    def clear_buffer(self):
        self.agent_reward_list = [[] for x in range(self.agent_num)]
        self.new_episode_lengths.clear()
        self.new_episode_rewards.clear()

    def get_episode_stats(self):
        rewards, lengths = None, None
        # If new episodes have been played return their lengths and rewards as well as the total number of episodes
        # played and the number of new episode since last readout
        if len(self.new_episode_rewards):
            rewards = self.new_episode_rewards.copy()
            lengths = self.new_episode_lengths.copy()
            self.total_episodes_played += len(self.new_episode_rewards)
            self.new_episode_lengths.clear()
            self.new_episode_rewards.clear()

        return lengths, rewards, self.total_episodes_played


@ray.remote
class GlobalLogger:
    """
    Global Logger
    Collects rewards and episode lengths from the actor's local loggers and sends them to Tensorboard.
    Furthermore, it tests if the conditions for a new model checkpoint are given.
    The Tensorboard files are stored in "./training/summaries" and can be viewed by opening a command line in this very
    directory and typing "tensorboard --logdir ."
    """
    def __init__(self, log_dir="./summaries",
                 tensorboard=True,
                 actor_num=1,
                 checkpoint_saving=True,
                 running_average_episodes=100,
                 behavior_clone_name=None):
        # Summary file path and logger creation time
        self.log_dir = log_dir
        self.creation_time = datetime.now()

        # Checkpoints
        self.checkpoint_saving = checkpoint_saving

        # Number of parallel actors
        self.actor_num = actor_num
        self.tensorboard = tensorboard

        # Memory of past rewards and the count of episodes played
        self.episode_reward_deque = [deque(maxlen=1000) for i in range(self.actor_num)]
        self.episode_length_deque = [deque(maxlen=1000) for i in range(self.actor_num)]
        self.episodes_played_per_actor = [0 for i in range(self.actor_num)]
        self.total_episodes_played = 0
        self.best_actor = 0
        self.average_rewards = [-10000 for i in range(self.actor_num)]
        self.new_episodes = 0
        self.behavior_clone_name = behavior_clone_name
        print("CLONE NAME", behavior_clone_name)

        # The best running average over 30 episodes
        self.best_running_average_reward = -10000
        self.last_save_time_step = 0
        self.running_average_episodes = running_average_episodes

        # Tensorboard Writer
        if self.tensorboard:
            self.tensorboard_writer = tf.summary.create_file_writer(log_dir)
            self.logger_dict = {}

    def get_elapsed_time(self):
        elapsed_time = datetime.now() - self.creation_time
        days, remainder = divmod(elapsed_time.seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return days, hours, minutes, seconds

    def get_best_running_average(self):
        return self.best_running_average_reward

    def append(self, episode_lengths, episode_rewards, total_number_of_episodes, actor_idx=0):
        if not episode_lengths:
            return
        # Append new episode rewards and lengths
        self.new_episodes += len(episode_rewards)
        self.episode_reward_deque[actor_idx].extend(episode_rewards)
        self.episode_length_deque[actor_idx].extend(episode_lengths)
        # Total episodes per agent and overall
        self.episodes_played_per_actor[actor_idx] = total_number_of_episodes
        self.total_episodes_played = np.sum(self.episodes_played_per_actor)
        # Average rewards per agent
        self.average_rewards[actor_idx] = \
            np.mean(list(self.episode_reward_deque[actor_idx])[-self.running_average_episodes:])
        self.best_actor = np.argmax(self.average_rewards)

        if self.tensorboard:
            self.log_dict({"Reward/Agent{:03d}Reward".format(actor_idx):
                               self.episode_reward_deque[actor_idx][-1],
                           "EpisodeLength/Agent{:03d}EpisodeLength".format(actor_idx):
                               self.episode_length_deque[actor_idx][-1]}, total_number_of_episodes, 1)

    def get_episode_stats(self):
        self.total_episodes_played = np.sum(self.episodes_played_per_actor)

        for reward in self.episode_reward_deque:
            if not len(reward):
                return 0, 0, self.total_episodes_played

        self.average_rewards = [np.mean(list(rewards)[-self.running_average_episodes:]) for rewards in self.episode_reward_deque]
        mean_reward = np.mean(self.average_rewards)
        mean_length = np.mean([np.mean(list(lengths)[-self.running_average_episodes:]) for lengths in self.episode_length_deque])
        self.new_episodes = 0

        return mean_length, mean_reward, self.total_episodes_played

    @ray.method(num_returns=3)
    def get_current_max_stats(self, average_num):
        max_agent_average_reward = np.max(self.average_rewards)
        length_list = list(self.episode_length_deque[self.best_actor])
        if len(length_list) < average_num:
            max_agent_average_length = 0
        else:
            max_agent_average_length = np.mean(list(self.episode_length_deque[self.best_actor])[-average_num:])
        return max_agent_average_length, max_agent_average_reward, self.total_episodes_played

    def get_new_sequence_length(self, sequence_length, training_step):
        # Only check for an updated sequence length every 100 training steps
        if training_step % 100 or not training_step:
            return None
        length_list = []
        # Append all episode lengths for all actors into one list
        for episode_lengths in self.episode_length_deque:
            length_list += list(episode_lengths)[-100:]
        # Sort the list from low to high and look at the length of the lower 20%. This means that 80% percent of the
        # recorded episodes are long enough to be sampled from for training. The others will be discarded.
        length_list.sort()
        new_sequence_length = length_list[int(0.25*len(length_list))]
        # Clamp the new sequence length between a minimum of 5 and a maximum of 80
        new_sequence_length = np.clip(new_sequence_length, 5, 80)
        # Only if the new sequence length differs more than 10 from the old recommend changing it.
        if np.abs(new_sequence_length - sequence_length) >= 10:
            print("New sequence length recommended!")
            print("Old sequence length: {}, new recommendation: {}, training step: {}".format(sequence_length,
                                                                                              new_sequence_length,
                                                                                              training_step))
            return new_sequence_length
        return None

    def check_checkpoint_condition(self):
        if self.checkpoint_saving:
            if self.total_episodes_played - self.last_save_time_step > self.running_average_episodes:
                max_average_reward = np.max(self.average_rewards)
                if max_average_reward > self.best_running_average_reward:
                    self.best_running_average_reward = max_average_reward
                    self.last_save_time_step = self.total_episodes_played
                    return True
                elif self.behavior_clone_name and self.total_episodes_played - 1000 >= self.last_save_time_step:
                    self.last_save_time_step = self.total_episodes_played
                    print("Periodical weight saving!")
                    return True
        return False

    def register_level_change(self, task_level_condition):
        if task_level_condition:
            self.best_running_average_reward = -10000
            self.last_save_time_step = self.total_episodes_played
            self.average_rewards = [-10000 for i in range(self.actor_num)]
            self.episode_reward_deque = [deque(maxlen=1000) for i in range(self.actor_num)]
            self.episode_length_deque = [deque(maxlen=1000) for i in range(self.actor_num)]

    def remove_old_checkpoints(self):
        def get_step_from_file(file):
            step = [f for f in file.split("_") if "Step" in f][0]
            step = int(step.replace("Step", ""))
            return step

        file_names = [f for f in os.listdir(self.log_dir) if f.endswith(".h5")]
        if len(file_names) <= 1:
            return False
        file_names.sort(key=get_step_from_file)
        highest_step = get_step_from_file(file_names[-1])
        for file_name in file_names:
            if str(highest_step) not in file_name:
                os.remove(os.path.join(self.log_dir, file_name))
        return True

    def log_dict(self, metrics, step, logging_frequency):
        if metrics:
            if step % logging_frequency == 0 or step == 1:
                for key, val in metrics.items():
                    self.log_scalar(key, val, step)

    def log_scalar(self, tag, value, step):
        with self.tensorboard_writer.as_default():
            tf.summary.scalar(tag, value, step)
            self.tensorboard_writer.flush()
