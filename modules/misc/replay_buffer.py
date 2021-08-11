#!/usr/bin/env python

"""
Advanced Replay Buffer

Stores a limited number of experience samples of form (s, a, r', s') from the environment.
These replays can later be used as replay buffer to train a neural network with random samples to improve the
performance. Is able to store n-step replays.

Created by Pascal Graf
Last edited 08.07.2020
"""

from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
prioritized_experience = namedtuple("PrioritizedExperience", field_names=["priority", "probability"])


class ReplayBuffer:
    def __init__(self,
                 capacity: int,
                 agent_num: int = 1,
                 n_steps: int = 1,
                 gamma: float = 1,
                 mode: str = "memory",
                 prioritized: bool = False,
                 prioritized_update_frequency: int = 5,
                 parameter_update_frequency: int = 500,
                 prediction_error_based: bool = False):
        self.buffer = deque(maxlen=capacity)
        self.min_size_reached = False
        self.n_steps = n_steps
        self.done_indices = set()
        self.agent_num = agent_num
        self.mode = mode

        # New Sample and Trajectory Counters
        self.new_training_samples = 0
        self.collected_trajectories = 0

        # TEST: Prioritized Experience Replay
        self.prioritized = prioritized
        self.prioritized_update_frequency = prioritized_update_frequency
        self.parameter_update_frequency = parameter_update_frequency
        self.prio_update = 0

        self.sampled_indices = None

        self.alpha = 0.5
        self.alpha_decay_rate = 0.99

        self.priority_sum_alpha = 0
        self.max_priority = 1
        self.max_weight = 1

        # TEST: Prediction error based buffer
        self.prediction_error_deque = deque(maxlen=10000)
        self.prediction_error_based = prediction_error_based

        if self.mode == "memory":
            self.state_deque = [deque(maxlen=n_steps+1) for x in range(self.agent_num)]
            self.action_deque = [deque(maxlen=n_steps+1) for x in range(self.agent_num)]
            self.reward_deque = [deque(maxlen=n_steps) for x in range(self.agent_num)]
        else:
            self.state_deque = [[] for x in range(self.agent_num)]
            self.action_deque = [[] for x in range(self.agent_num)]
            self.reward_deque = [[] for x in range(self.agent_num)]
        if prioritized:
            self.prioritized_deque = deque(maxlen=capacity)

        self.gamma = gamma
        self.gamma_list = [gamma ** n for n in range(n_steps)]

    def __len__(self):
        return len(self.buffer)

    def calculate_discounted_return(self, rewards):
        disc_return = []
        sum_reward = 0.0

        for r in reversed(rewards):
            sum_reward *= self.gamma
            sum_reward += r
            disc_return.append(sum_reward)
        return list(reversed(disc_return))

    def add_new_steps(self, states, rewards, ids, exclude_ids=[], actions=None, step_type='decision', add_to_done=False):
        for idx, agent_id in enumerate(ids):
            if agent_id in self.done_indices or agent_id in exclude_ids:
                continue
            state_component_list = []
            for state_component in states:
                state_component_list.append(state_component[idx])
            self.state_deque[agent_id].append(state_component_list)
            self.reward_deque[agent_id].append(rewards[idx])
            if np.all(actions) is not None:
                self.action_deque[agent_id].append(actions[idx])
            else:
                self.action_deque[agent_id].append(None)

            # If "done" append the remaining deque steps to the replay buffer
            # and clear the deques afterwards
            if step_type == 'terminal':
                if len(self.state_deque[agent_id]) > 1:
                    if self.mode == "memory":
                        for n in range(self.n_steps):
                            # Calculate the discounted reward for each value in the reward deque
                            discounted_reward = [r * g for r, g in zip(self.reward_deque[agent_id], self.gamma_list)]
                            self.append(self.state_deque[agent_id][0], self.action_deque[agent_id][0],
                                        np.sum(discounted_reward), self.state_deque[agent_id][-1],
                                        True)
                            # Append placeholder values to each deque
                            self.state_deque[agent_id].append(self.state_deque[agent_id][-1])
                            self.action_deque[agent_id].append(None)
                            self.reward_deque[agent_id].append(0)
                    else:
                        # discounted_return = self.calculate_discounted_return(self.reward_deque[agent_id])
                        for i, (state, action, reward) in enumerate(zip(self.state_deque[agent_id][:-1],
                                                                        self.action_deque[agent_id][:-1],
                                                                        self.reward_deque[agent_id][1:])):
                            if i == len(self.state_deque[agent_id][:-1])-1:
                                self.append(state, action, reward, state, True)
                            else:
                                self.append(state, action, reward, state, False)
                        self.collected_trajectories += 1
                if self.mode == "trajectory" or self.agent_num == 1 or add_to_done:
                    self.done_indices.add(agent_id)

                # Clear the deques
                self.state_deque[agent_id].clear()
                self.action_deque[agent_id].clear()
                self.reward_deque[agent_id].clear()

            if len(self.state_deque[agent_id]) == self.n_steps+1 and self.mode == "memory":
                # Calculate the discounted reward for each value in the reward deque
                discounted_reward = [r * g for r, g in zip(self.reward_deque[agent_id], self.gamma_list)]
                self.append(self.state_deque[agent_id][0], self.action_deque[agent_id][0],
                            np.sum(discounted_reward), self.state_deque[agent_id][-1],
                            False)

    def append(self, s, a, r, next_s, done):
        if self.prioritized:
            if np.any(self.sampled_indices):
                self.sampled_indices -= 1
                for idx, s_idx in enumerate(self.sampled_indices):
                    if s_idx < 0:
                        self.sampled_indices[idx] = np.random.randint(0, len(self.buffer))
            # If the maximum buffer capacity is reached
            if len(self.buffer) == self.buffer.maxlen:
                # Update the sum of priorities
                overwritten_experience = self.prioritized_deque[0]
                self.priority_sum_alpha -= overwritten_experience.probability ** self.alpha
                # If the overwritten experience has the maximum priority perform a search for the new max
                if overwritten_experience.priority == self.max_priority:
                    self.prioritized_deque[0].priority = 0
                    self.max_priority = max([p.priority for p in self.prioritized_deque])
            # Set the new priority to the max priority and add to the priority sum
            priority = self.max_priority
            self.priority_sum_alpha += priority ** self.alpha
            probability = priority ** self.alpha / self.priority_sum_alpha
            # Append the new priority and probability
            self.prioritized_deque.append(prioritized_experience(priority, probability))
        self.buffer.append({"state": s, "action": a, "reward": r, "next_state": next_s, "done": done})
        self.new_training_samples += 1

    def check_training_condition(self, trainer_configuration):
        if self.mode == "memory":
            if len(self.buffer) > trainer_configuration['ReplayMinSize']:
                self.min_size_reached = True
            if self.new_training_samples >= trainer_configuration['TrainingInterval'] and self.min_size_reached:
                return True
        else:
            if self.collected_trajectories >= trainer_configuration['TrajectoryNum']:
                return True
        return False

    def check_reset_condition(self):
        return len(self.done_indices) >= self.agent_num

    def check_sample_error(self, prediction_errors):
        for e in prediction_errors:
            self.prediction_error_deque.append(e)
        if len(self.prediction_error_deque) >= self.prediction_error_deque.maxlen:
            prediction_error_list = list(self.prediction_error_deque)
            prediction_error_list.sort()
            threshold = prediction_error_list[int(0.5*self.prediction_error_deque.maxlen)]

            for idx, e in enumerate(prediction_errors):
                if e < threshold:
                    del self.buffer[-len(prediction_errors)+idx]

    def sample(self,
               batch_size: int,
               throwaway: bool = False,
               random_samples: bool = True):
        indices = []
        if self.mode == "memory":
            if random_samples:
                if self.prioritized:
                    if self.prio_update >= self.prioritized_update_frequency:
                        self.prio_update = 0
                    if self.prio_update == 0:
                        self.sampled_indices = np.random.choice(len(self.buffer),
                                                                batch_size*self.prioritized_update_frequency,
                                                                p=[p.probability for p in self.prioritized_deque],
                                                                replace=True)
                    indices = self.sampled_indices[self.prio_update*batch_size: (self.prio_update+1)*batch_size]
                    self.prio_update += 1
                else:
                    indices = np.random.choice(len(self.buffer), batch_size, replace=False)
                replay_batch = [self.buffer[idx] for idx in indices]
                self.new_training_samples = 0
            else:
                return [self.buffer[-self.new_training_samples+idx] for idx in range(self.new_training_samples)]
        else:
            replay_batch = [transition for transition in self.buffer]
            self.buffer.clear()
            self.collected_trajectories = 0
            if throwaway:
                self.state_deque = [[] for x in range(self.agent_num)]
                self.action_deque = [[] for x in range(self.agent_num)]
                self.reward_deque = [[] for x in range(self.agent_num)]
        copy_by_val_replay_batch = deepcopy(replay_batch)
        return copy_by_val_replay_batch, indices

    def update_parameters(self, training_step):
        if self.prioritized:
            if training_step % self.parameter_update_frequency == 0:
                self.alpha *= self.alpha_decay_rate

            self.priority_sum_alpha = 0
            sum_prob_before = 0
            for element in self.prioritized_deque:
                sum_prob_before += element.probability
                self.priority_sum_alpha += element.priority**self.alpha
            sum_prob_after = 0
            for index, element in enumerate(self.prioritized_deque):
                probability = element.priority ** self.alpha / self.priority_sum_alpha
                sum_prob_after += probability
                self.prioritized_deque[index] = prioritized_experience(element.priority, probability)

    def update_priorities(self, losses, indices):
        if self.prioritized:
            for loss, index in zip(losses, indices):
                index = int(index[0])
                updated_priority = loss
                if updated_priority > self.max_priority:
                    self.max_priority = updated_priority

                old_priority = self.prioritized_deque[index].priority
                self.priority_sum_alpha += updated_priority ** self.alpha - old_priority**self.alpha
                updated_probability = loss**self.alpha / self.priority_sum_alpha
                self.prioritized_deque[index] = prioritized_experience(updated_priority, updated_probability)

    def print_plot_buffer_sample(self):
        indices = np.random.choice(len(self.buffer), 10, replace=False)
        replay_batch = [self.buffer[idx] for idx in indices]
        for sample in replay_batch:
            for observation, next_observation in zip(sample["state"], sample["next_state"]):
                if len(observation.shape) == 1:
                    print("Observation Vector:" + observation)
                    print("Next Observation Vector:" + next_observation)
                elif len(observation.shape) == 3:
                    if observation.shape[2] == 4:
                        fig, axs = plt.subplots(4)
                        for idx, ax in enumerate(axs):
                            ax.imshow(observation[:, :, idx])
                        fig2, axs2 = plt.subplots(4)
                        for idx2, ax2 in enumerate(axs2):
                            ax2.imshow(next_observation[:, :, idx2])
                        plt.show()
                    elif observation.shape[2] == 12:
                        fig, axs = plt.subplots(4)
                        for idx, ax in enumerate(axs):
                            ax.imshow(observation[:, :, idx*3:(idx+1)*3])
                        plt.savefig("plots/observations/obs.png")
                        plt.close(fig)
                        fig2, axs2 = plt.subplots(4)
                        for idx2, ax2 in enumerate(axs2):
                            ax2.imshow(next_observation[:, :, idx2*3:(idx2+1)*3])
                        plt.savefig("plots/observations/next_obs.png")
                        plt.close(fig2)
                    elif observation.shape[2] == 3:
                        plt.imshow(observation)
                        plt.imshow(next_observation)
                        plt.show()
                break


