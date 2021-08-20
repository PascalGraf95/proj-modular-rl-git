#!/usr/bin/env python

from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
prioritized_experience = namedtuple("PrioritizedExperience", field_names=["priority", "probability"])


class FIFOBuffer:
    """

    """
    def __init__(self,
                 capacity: int,
                 agent_num: int = 1,
                 n_steps: int = 1,
                 gamma: float = 1,
                 store_trajectories: bool = False,
                 prioritized: bool = False,
                 prioritized_update_frequency: int = 5,
                 parameter_update_frequency: int = 500):

        # Initialize actual buffer with defined capacity
        self.buffer = deque(maxlen=capacity)
        # Flag to indicate if training threshold has been reached
        self.min_size_reached = False
        # n-Step reward sum
        self.n_steps = n_steps
        # Keep track of which agent's episodes have ended in the simulation (relevant if multiple)
        self.done_agents = set()
        self.agent_num = agent_num
        # If set to true, only whole trajectories are written into the buffer, otherwise they are written step-by-step
        # which results in arbitrary order for multiple agents in one environment.
        self.store_trajectories = store_trajectories

        # New sample and trajectory counters
        self.new_training_samples = 0
        self.collected_trajectories = 0

        # Prioritized Experience Replay
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

        if prioritized:
            self.prioritized_deque = deque(maxlen=capacity)

        # Deque to enable n-step reward calculation
        self.state_deque = [deque(maxlen=n_steps+1) for x in range(self.agent_num)]
        self.action_deque = [deque(maxlen=n_steps+1) for x in range(self.agent_num)]
        self.reward_deque = [deque(maxlen=n_steps) for x in range(self.agent_num)]

        # Temporal buffer for storing trajectories
        self.temp_agent_buffer = [[] for x in range(self.agent_num)]

        # Discount factor
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

    def add_new_steps(self, states, rewards, ids, exclude_ids=[], actions=None,
                      step_type='decision'):
        # Iterate through all available agents
        for idx, agent_id in enumerate(ids):
            # Don't add experiences of agents whose episode already ended or which are manually excluded
            if agent_id in self.done_agents or agent_id in exclude_ids:
                continue
            # Combine all observations of one agent in one list
            state_component_list = []
            for state_component in states:
                state_component_list.append(state_component[idx])
            # Add observed state, reward and action to the deque
            self.state_deque[agent_id].append(state_component_list)
            self.reward_deque[agent_id].append(rewards[idx])
            if np.all(actions) is not None:
                self.action_deque[agent_id].append(actions[idx])
            else:
                self.action_deque[agent_id].append(None)

            # If "done" append the remaining deque steps to the replay buffer
            # and clear the deques afterwards
            if step_type == 'terminal':
                # Only add the collected experiences to the replay buffer if the episode was at least 2 steps long.
                if len(self.state_deque[agent_id]) > 1:
                    for n in range(self.n_steps):
                        # Calculate the discounted reward for each value in the reward deque
                        discounted_reward = [r * g for r, g in zip(self.reward_deque[agent_id], self.gamma_list)]
                        # Add the experience to the temporal agent buffer
                        self.temp_agent_buffer.append([self.state_deque[agent_id][0], self.action_deque[agent_id][0],
                                                       np.sum(discounted_reward), self.state_deque[agent_id][-1],
                                                       True])
                        # Append placeholder values to each deque
                        self.state_deque[agent_id].append(self.state_deque[agent_id][-1])
                        self.action_deque[agent_id].append(None)
                        self.reward_deque[agent_id].append(0)

                # Clear the deques
                self.state_deque[agent_id].clear()
                self.action_deque[agent_id].clear()
                self.reward_deque[agent_id].clear()

                # Write the collected data to the actual replay buffer.
                for experience in self.temp_agent_buffer[agent_id]:
                    self.append(*experience)
                    self.temp_agent_buffer[idx].clear()
                self.collected_trajectories += 1

                if self.store_trajectories or self.agent_num == 1:
                    self.done_agents.add(agent_id)

            # Write the deque data to the temporal buffer
            if len(self.state_deque[agent_id]) == self.n_steps+1:
                # Calculate the discounted reward for each value in the reward deque
                discounted_reward = [r * g for r, g in zip(self.reward_deque[agent_id], self.gamma_list)]
                self.temp_agent_buffer.append([self.state_deque[agent_id][0], self.action_deque[agent_id][0],
                                               np.sum(discounted_reward), self.state_deque[agent_id][-1],
                                               True])

            # Write the collected data to the actual replay buffer if not storing whole trajectories.
            if not self.store_trajectories:
                for experience in self.temp_agent_buffer[agent_id]:
                    self.append(*experience)
                self.temp_agent_buffer[idx].clear()

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
        if not self.store_trajectories:
            if len(self.buffer) > trainer_configuration['ReplayMinSize']:
                self.min_size_reached = True
            if self.new_training_samples >= trainer_configuration['TrainingInterval'] and self.min_size_reached:
                return True
        else:
            if self.collected_trajectories >= trainer_configuration['TrajectoryNum']:
                return True
        return False

    def check_reset_condition(self):
        return len(self.done_agents) >= self.agent_num

    def append_list(self, samples):
        self.buffer.extend(samples)

    def sample(self,
               batch_size: int,
               reset_buffer: bool = False,
               random_samples: bool = True):
        indices = []
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
        else:
            if batch_size == -1:
                replay_batch = [transition for transition in self.buffer]
            else:
                replay_batch = [self.buffer[-self.new_training_samples+idx] for idx in range(self.new_training_samples)]
        if reset_buffer:
            self.buffer.clear()
            self.state_deque = [[] for x in range(self.agent_num)]
            self.action_deque = [[] for x in range(self.agent_num)]
            self.reward_deque = [[] for x in range(self.agent_num)]

        self.collected_trajectories = 0
        self.new_training_samples = 0

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


