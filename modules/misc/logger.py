import tensorflow as tf
import numpy as np
import os
from collections import deque
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
        # Number of agents in the environment + list of rewards over the current episode, each agent in the environment
        # gets its own list
        self.agent_num = agent_num
        self.agent_reward_list = [[] for x in range(self.agent_num)]
        # Lists for rewards and episode lengths since last readout
        self.new_episode_rewards = []
        self.new_episode_lengths = []

    def track_episode(self, rewards, agent_ids, step_type='decision'):
        # For each agent in the environment, track the rewards of the current episode
        for idx, agent_id in enumerate(agent_ids):
            # Append the respective reward to the respective agent in the nested list
            self.agent_reward_list[agent_id].append(rewards[idx])
            # If the current state is terminal add the summed rewards as well as the length of the current agent's
            # episode to the buffer and reset its list.
            if step_type == 'terminal':
                self.new_episode_rewards.append(np.sum(self.agent_reward_list[agent_id]))
                self.new_episode_lengths.append(len(self.agent_reward_list[agent_id]))
                self.agent_reward_list[agent_id].clear()

    def clear_buffer(self):
        # Reset the buffer for all agents.
        self.agent_reward_list = [[] for x in range(self.agent_num)]
        self.new_episode_lengths.clear()
        self.new_episode_rewards.clear()

    def get_episode_stats(self):
        # Return the latest stats to the global logger.
        # If new episodes have been played return their lengths and rewards as well as the total number of episodes
        # played and the number of new episode since last readout
        rewards, lengths = None, None
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
                 running_average_episodes=100,
                 periodic_model_saving=False,
                 device: str = '/cpu:0'):
        # Set CPU as device to run Tensorboard
        self.device = device

        # Summary file path and logger creation time
        self.log_dir = log_dir
        self.creation_time = datetime.now()

        # Number of parallel actors
        self.actor_num = actor_num
        self.tensorboard = tensorboard
        self.last_logged_dict = {}

        # Memory of the past 1000 rewards and episode lengths
        self.episode_reward_deque = [deque(maxlen=1000) for i in range(self.actor_num)]
        self.episode_length_deque = [deque(maxlen=1000) for i in range(self.actor_num)]
        # The count of episodes each actor played as well as the total number of episodes played.
        self.episodes_played_per_actor = [0 for i in range(self.actor_num)]
        self.total_episodes_played = 0
        # Number of episodes played since the stats have been read out last.
        self.new_episodes = 0

        # Index of the actor with the highest running average reward
        self.best_actor = 0
        # Running average rewards of all actors
        self.average_rewards = [-10000 for i in range(self.actor_num)]
        # The best running average over "running_average_episodes" episodes
        self.best_running_average_reward = -10000
        self.running_average_episodes = running_average_episodes
        # Total episode number at which the network weights have been saved last.
        self.last_save_time_step = 0
        # If true, new models are saved periodically independent of the agents' rewards
        self.periodic_model_saving = periodic_model_saving

        # Tensorboard Writer
        if self.tensorboard:
            with tf.device(self.device):
                self.tensorboard_writer = tf.summary.create_file_writer(log_dir)
                self.logger_dict = {}

    def get_elapsed_time(self):
        # Returns the elapsed time since creation of this logger at the beginning of the training process.
        elapsed_time = datetime.now() - self.creation_time
        days, remainder = divmod(elapsed_time.seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        return days, hours, minutes, seconds

    def get_best_running_average(self):
        # Returns the best running average which will be updated on each call of the checkpoint condition function.
        return self.best_running_average_reward

    def append(self, episode_lengths, episode_rewards, total_number_of_episodes, actor_idx=0):
        # Append episodes from the local buffers to this global buffer.
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

    @ray.method(num_returns=3)
    def get_current_max_stats(self, average_num):
        # Returns the running average stats (episode length & reward) of the best actor and the total number of
        # episodes played.
        max_agent_average_reward = self.average_rewards[self.best_actor]
        length_list = list(self.episode_length_deque[self.best_actor])
        if len(length_list) < average_num:
            max_agent_average_length = 0
        else:
            max_agent_average_length = np.mean(list(self.episode_length_deque[self.best_actor])[-average_num:])
        return max_agent_average_length, max_agent_average_reward, self.total_episodes_played

    def get_total_episodes(self):
        return self.total_episodes_played

    @ray.method(num_returns=2)
    def get_new_sequence_length(self, sequence_length, burn_in, training_step):
        # Only check for an updated sequence length every 100 training steps
        if training_step % 100 or not training_step:
            return None, None
        length_list = []
        # Append all episode lengths for all actors into one list
        for episode_lengths in self.episode_length_deque:
            length_list += list(episode_lengths)[-100:]
        # Sort the list from low to high and look at the length of the lower 20%. This means that 80% of the
        # recorded episodes are long enough to be sampled from for training. The others will be discarded.
        length_list.sort()
        new_sequence_length = length_list[int(0.20*len(length_list))]
        # Clamp the new sequence length between a minimum of 5 and a maximum of 80
        new_sequence_length = np.clip(new_sequence_length, 5, 80)
        # Only if the new sequence length differs more than 10 from the old recommend changing it.
        if np.abs(new_sequence_length - sequence_length) >= 10:
            new_burn_in = 0
            if burn_in:
                if new_sequence_length > 15:
                    new_burn_in = new_sequence_length // 4
            print("New sequence length recommended!")
            print("Old sequence length: {}, new recommendation: {}, "
                  "Old burn in: {}, new burn in: {}, training step: {}".format(sequence_length,
                                                                               new_sequence_length,
                                                                               burn_in,
                                                                               new_burn_in,
                                                                               training_step))
            return new_sequence_length, new_burn_in
        return None, None

    def check_checkpoint_condition(self, training_step):
        if self.periodic_model_saving:
            if training_step - self.last_save_time_step >= 10000:
                self.last_save_time_step = training_step
                print("Periodic Model Saving at Step {}".format(training_step))
                return True
            if training_step - self.last_save_time_step > 1000 and self.total_episodes_played > 10:
                max_average_reward = self.average_rewards[self.best_actor]
                if max_average_reward > self.best_running_average_reward:
                    self.best_running_average_reward = max_average_reward
                    self.last_save_time_step = training_step
                    return True

        # Returns true if the conditions for a new checkpoint are met.
        if self.total_episodes_played - self.last_save_time_step > 10 and self.total_episodes_played > 100:
            max_average_reward = self.average_rewards[self.best_actor]
            if max_average_reward > self.best_running_average_reward:
                self.best_running_average_reward = max_average_reward
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

    def log_dict(self, metrics, step, logging_frequency):
        if metrics:
            if step % logging_frequency == 0 or step == 1:
                for key, val in metrics.items():
                    if self.last_logged_dict.get(key) != step:
                        self.log_scalar(key, val, step)
                    self.last_logged_dict[key] = step

    def log_scalar(self, tag, value, step):
        with tf.device(self.device):
            with self.tensorboard_writer.as_default():
                tf.summary.scalar(tag, value, step)
                self.tensorboard_writer.flush()
