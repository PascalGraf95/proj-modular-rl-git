import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import shutil
from collections import deque
from tensorflow.python.summary.summary_iterator import summary_iterator
from datetime import datetime

"""
Logger

Creates a Tensorboard Logger for scalar values or running average values.
Methods to append new values are provided.

Created by Pascal Graf
Last edited 29.01.2021
"""


class LocalLogger:
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


class GlobalLogger:
    def __init__(self, log_dir="./summaries",
                 tensorboard=True,
                 actor_num=1,
                 checkpoint_saving=True,
                 running_average_episodes=30):
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

    def append(self, episode_rewards, episode_lengths, total_number_of_episodes, actor_idx=0):
        self.episode_reward_deque[actor_idx].extend(episode_rewards)
        self.episode_length_deque[actor_idx].extend(episode_lengths)
        self.episodes_played_per_actor[actor_idx] = total_number_of_episodes
        if self.tensorboard:
            self.log_dict({"Performance/Rewards/Agent{:03d}Reward".format(actor_idx):
                               self.episode_reward_deque[actor_idx][-1],
                           "Performance/EpisodeLengths/Agent{:03d}Length".format(actor_idx):
                               self.episode_length_deque[actor_idx][-1]}, total_number_of_episodes)

    def get_episode_stats(self):
        self.average_rewards = [np.mean(list(rewards)[-self.running_average_episodes:]) for rewards in self.episode_reward_deque]
        mean_reward = np.mean(self.average_rewards)
        mean_length = np.mean([np.mean(list(lengths)[-self.running_average_episodes:]) for lengths in self.episode_length_deque])
        self.total_episodes_played = np.sum(self.episodes_played_per_actor)

        return mean_length, mean_reward, self.total_episodes_played

    def get_current_max_stats(self, average_num):
        max_agent_average = np.mean(list(self.episode_reward_deque[self.best_actor])[-average_num:])
        return max_agent_average, self.total_episodes_played

    def check_checkpoint_condition(self):
        if self.checkpoint_saving:
            if self.total_episodes_played - self.last_save_time_step > self.running_average_episodes:
                max_average_reward = np.max(self.average_rewards)
                if max_average_reward > self.best_running_average_reward:
                    self.episode_reward_deque = [deque(maxlen=1000) for i in range(self.actor_num)]
                    self.episode_length_deque = [deque(maxlen=1000) for i in range(self.actor_num)]
                    self.best_running_average_reward = max_average_reward
                    self.best_actor = np.argmax(self.average_rewards)
                    self.last_save_time_step = self.total_episodes_played
                    return True
        return False

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

    def log_dict(self, metrics, step):
        for key, val in metrics.items():
            self.log_scalar(key, val, step)

    def log_scalar(self, tag, value, step):
        with self.tensorboard_writer.as_default():
            tf.summary.scalar(tag, value, step)
