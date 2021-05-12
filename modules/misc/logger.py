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


class Logger:
    def __init__(self, log_dir="./summaries",
                 tensorboard=True,
                 agent_num=1,
                 early_stopping=False,
                 checkpoint_saving=True,
                 mode="memory"):
        self.log_dir = log_dir

        # Early Stopping / Checkpoints
        self.early_stopping = early_stopping
        self. checkpoint_saving = checkpoint_saving
        self.episode_reward_memory = deque(maxlen=10000)
        self.episodes_played_memory = 0

        self.best_running_average_reward = -10000
        self.time_steps_since_last_save = 0

        # Done Indices
        self.done_indices = set()
        self.mode = mode

        # Logger Creation Time
        self.creation_time = datetime.now()
        # Tensorboard Writer
        if tensorboard:
            self.tensorboard_writer = tf.summary.create_file_writer(log_dir)
            self.logger_dict = {}
        # Local Tracker
        self.agent_num = agent_num
        self.agent_reward_list = [[] for x in range(self.agent_num)]
        self.episode_rewards = []
        self.episode_lengths = []

    def get_elapsed_time(self):
        elapsed_time = datetime.now() - self.creation_time
        days, remainder = divmod(elapsed_time.seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return days, hours, minutes, seconds

    def track_episode(self, rewards, agent_ids, exclude_ids=[], step_type='decision', add_to_done=False):
        for idx, agent_id in enumerate(agent_ids):
            if agent_id in self.done_indices or agent_id in exclude_ids:
                continue
            self.agent_reward_list[agent_id].append(rewards[idx])
            if step_type == 'terminal':
                self.episode_rewards.append(np.sum(self.agent_reward_list[agent_id]))
                self.episode_lengths.append(len(self.agent_reward_list[agent_id]))
                self.agent_reward_list[agent_id].clear()
                if self.mode == "trajectory" or add_to_done:
                    self.done_indices.add(agent_id)

    def clear_buffer(self):
        self.done_indices.clear()
        self.agent_reward_list = [[] for x in range(self.agent_num)]
        self.episode_lengths.clear()
        self.episode_rewards.clear()

    def get_episode_stats(self, track_stats=False, throwaway=False):
        if throwaway:
            self.clear_buffer()
        mean_reward, mean_length = None, None
        new_episodes = len(self.episode_rewards)
        if len(self.episode_rewards):
            mean_reward = np.mean(self.episode_rewards)
            print(mean_reward);
            mean_length = np.mean(self.episode_lengths)

            if track_stats:
                self.episodes_played_memory += len(self.episode_rewards)
                self.episode_reward_memory += self.episode_rewards
                self.time_steps_since_last_save += len(self.episode_lengths)
            self.episode_lengths.clear()
            self.episode_rewards.clear()
        episodes = self.episodes_played_memory

        return mean_length, mean_reward, episodes, new_episodes

    def check_early_stopping_condition(self):
        if self.early_stopping:
            indices = 30
            if len(self.episode_reward_memory) < indices:
                return False
            # If previous conditions are met, calculate the coefficient of variation (stddev / mean)
            var_coeff = np.std(list(self.episode_reward_memory)[-indices:]) / \
                        np.mean(list(self.episode_reward_memory)[-indices:])
            if np.abs(var_coeff) < 0.01:
                running_average_reward = np.mean(list(self.episode_reward_memory)[-indices:])
                if running_average_reward >= self.best_running_average_reward:
                    self.best_running_average_reward = running_average_reward
                    return True
        return False

    def check_checkpoint_condition(self):
        if self.checkpoint_saving:
            indices = 30
            if self.time_steps_since_last_save > indices:
                running_average_reward = np.mean(list(self.episode_reward_memory)[-indices:])
                if running_average_reward > self.best_running_average_reward:
                    self.best_running_average_reward = running_average_reward
                    self.time_steps_since_last_save = 0
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
        # if tag not in self.logger_dict.keys():
        #     self.logger_dict[tag] = [], []
        # self.logger_dict[tag][0].append(value)
        # self.logger_dict[tag][1].append(step)

    def log_running_average(self, tag, run_avg_len=20):
        tag_len = len(self.logger_dict[tag][0])
        tag_vals = self.logger_dict[tag][0][-int(min(tag_len, run_avg_len)):]
        with self.tensorboard_writer.as_default():
            tf.summary.scalar("Avg"+tag, np.mean(tag_vals), self.logger_dict[tag][1][-1])

    def get_running_average(self, tag, run_avg_len=20):
        tag_len = len(self.logger_dict[tag][0])
        tag_vals = self.logger_dict[tag][0][-int(min(tag_len, run_avg_len)):]
        return np.mean(tag_vals)


if __name__ == '__main__':
    logger = Logger(r"C:\PGraf\Arbeit\RL\ModularRL\training\summaries\200824_100706_DDPG_RobotTest", tensorboard=False)
    logger.remove_old_checkpoints()