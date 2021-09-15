#!/usr/bin/env python

from collections import namedtuple, deque
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import ray
prioritized_experience = namedtuple("PrioritizedExperience", field_names=["priority", "probability"])


# Modified from: https://pylessons.com/CartPole-PER/
class SumTree:
    data_pointer = 0

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity
        # Generate the tree with all nodes values = 0
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)
        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def add(self, priority, data):
        # Which index to we want to put the experience, leaves will be filled from left to right
        idx = self.data_pointer + self.capacity - 1
        # Update data
        self.data[self.data_pointer] = data
        # Update the leaf
        self.update(idx, priority)
        # Add 1 to the data pointer
        self.data_pointer += 1
        # If maximum capacity is reached, go back to the first index
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        # Keep track of the number of entries in the buffer
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        # Then propagate the change through the tree
        # self._propagate(idx, change)
        # TODO: Test while loop vs. recursive function time
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get(self, v):
        parent_index = 0
        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            # If the bottom level has been reached, end the search
            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            # Else reach downward one level
            else:
                if v <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    v -= self.tree[left_child_index]
                    parent_index = right_child_index

        # Convert from leaf index to data index (which has half the capacity)
        data_index = leaf_index - self.capacity + 1
        # Return leaf index, sample priority and the actual data
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def total_priority(self):
        return self.tree[0]


@ray.remote(num_cpus=1)
class PrioritizedBuffer:
    per_e = 0.001  # Avoid samples to have 0 probability of being taken
    per_a = 0.6  # tradeoff between taking only experiences with high priority vs. uniform sampling
    per_beta = 0.4  # importance-sampling, from initial value increasing to 1
    per_beta_increment_per_sampling = 0.001

    def __init__(self, capacity: int):

        # Initialize sum tree
        self.tree = SumTree(capacity)
        self.capacity = capacity

        # Flag to indicate if training threshold has been reached
        self.min_size_reached = False

        # New sample and trajectory counters
        self.new_training_samples = 0
        self.collected_trajectories = 0
        self.steps_without_training = 0

    def __len__(self):
        return self.tree.n_entries

    def reset(self):
        self.tree = SumTree(self.capacity)

    def check_training_condition(self, trainer_configuration):
        if self.tree.n_entries > trainer_configuration['ReplayMinSize']:
            self.min_size_reached = True
        self.steps_without_training += 1
        if self.steps_without_training >= trainer_configuration['TrainingInterval'] and self.min_size_reached:
            self.steps_without_training = 0
            return True
        return False

    def _getPriority(self, error):
        return (error + self.per_e) ** self.per_a

    def append(self, s, a, r, next_s, done, error):
        priority = self._getPriority(error)
        self.tree.add(priority, {"state": s, "action": a, "reward": r, "next_state": next_s, "done": done})
        self.new_training_samples += 1

    def append_list(self, samples, errors):
        for sample, error in zip(samples, errors):
            priority = self._getPriority(error)
            self.tree.add(priority, sample)
        self.new_training_samples += len(samples)

    def sample(self, batch_size: int):
        batch = []
        batch_indices = np.empty((batch_size,), dtype=np.int32)

        # Priority segment = total priority / number of samples
        priority_segment = self.tree.total_priority() / batch_size
        # priorities = []
        self.per_beta = np.min([1., self.per_beta + self.per_beta_increment_per_sampling])

        for i in range(batch_size):
            # A value is uniformly sampled from each range
            a, b = priority_segment * i, priority_segment * (i+1)
            value = np.random.uniform(a, b)
            index, priority, data = self.tree.get(value)
            # priorities.append(priority)
            batch.append(data)
            batch_indices[i] = index

        # sampling_probabilities = priorities / self.tree.total_priority()
        # is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        # is_weight /= is_weight.max()
        # TODO: Implement Importance Sampling

        self.collected_trajectories = 0
        self.new_training_samples = 0

        copy_by_val_replay_batch = deepcopy(batch)
        return copy_by_val_replay_batch, batch_indices  # , is_weight

    def update(self, indices, errors):
        for idx, error in zip(indices, errors):
            priority = self._getPriority(error)
            self.tree.update(idx, priority)


@ray.remote(num_cpus=1)
class FIFOBuffer:
    """

    """
    def __init__(self,
                 capacity: int,
                 agent_num: int = 1,
                 n_steps: int = 1,
                 gamma: float = 1,
                 store_trajectories: bool = False):

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

        self.sampled_indices = None

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
                        self.temp_agent_buffer[agent_id].append([self.state_deque[agent_id][0], self.action_deque[agent_id][0],
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
                self.temp_agent_buffer[agent_id].clear()
                self.collected_trajectories += 1

                if self.store_trajectories or self.agent_num == 1:
                    self.done_agents.add(agent_id)

            # Write the deque data to the temporal buffer
            if len(self.state_deque[agent_id]) == self.n_steps+1:
                # Calculate the discounted reward for each value in the reward deque
                discounted_reward = [r * g for r, g in zip(self.reward_deque[agent_id], self.gamma_list)]
                self.temp_agent_buffer[agent_id].append([self.state_deque[agent_id][0], self.action_deque[agent_id][0],
                                                         np.sum(discounted_reward), self.state_deque[agent_id][-1],
                                                         False])

            # Write the collected data to the actual replay buffer if not storing whole trajectories.
            if not self.store_trajectories:
                for experience in self.temp_agent_buffer[agent_id]:
                    self.append(*experience)
                self.temp_agent_buffer[idx].clear()

    def append(self, s, a, r, next_s, done):
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

    def update(self, indices, errors):
        pass

    def append_list(self, samples):
        self.buffer.extend(samples)
        self.new_training_samples += len(samples)

    def sample(self,
               batch_size: int,
               reset_buffer: bool = False,
               random_samples: bool = True):
        indices = []
        if random_samples:
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
            replay_batch = [self.buffer[idx] for idx in indices]
        else:
            if batch_size == -1:
                replay_batch = [transition for transition in self.buffer]
            else:
                replay_batch = [self.buffer[-self.new_training_samples+idx] for idx in range(self.new_training_samples)]
        if reset_buffer:
            self.buffer.clear()
            self.state_deque = [deque(maxlen=self.n_steps+1) for x in range(self.agent_num)]
            self.action_deque = [deque(maxlen=self.n_steps+1) for x in range(self.agent_num)]
            self.reward_deque = [deque(maxlen=self.n_steps) for x in range(self.agent_num)]

        self.collected_trajectories = 0
        self.new_training_samples = 0

        copy_by_val_replay_batch = deepcopy(replay_batch)
        return copy_by_val_replay_batch, indices


class LocalFIFOBuffer:
    """

    """
    def __init__(self,
                 capacity: int,
                 agent_num: int = 1,
                 n_steps: int = 1,
                 gamma: float = 1,
                 store_trajectories: bool = False):

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

        self.sampled_indices = None

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
                        self.temp_agent_buffer[agent_id].append([self.state_deque[agent_id][0], self.action_deque[agent_id][0],
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
                self.temp_agent_buffer[agent_id].clear()
                self.collected_trajectories += 1

                if self.store_trajectories or self.agent_num == 1:
                    self.done_agents.add(agent_id)

            # Write the deque data to the temporal buffer
            if len(self.state_deque[agent_id]) == self.n_steps+1:
                # Calculate the discounted reward for each value in the reward deque
                discounted_reward = [r * g for r, g in zip(self.reward_deque[agent_id], self.gamma_list)]
                self.temp_agent_buffer[agent_id].append([self.state_deque[agent_id][0], self.action_deque[agent_id][0],
                                                         np.sum(discounted_reward), self.state_deque[agent_id][-1],
                                                         False])

            # Write the collected data to the actual replay buffer if not storing whole trajectories.
            if not self.store_trajectories:
                for experience in self.temp_agent_buffer[agent_id]:
                    self.append(*experience)
                self.temp_agent_buffer[idx].clear()

    def append(self, s, a, r, next_s, done):
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

    def update(self, indices, errors):
        pass

    def append_list(self, samples):
        self.buffer.extend(samples)
        self.new_training_samples += len(samples)

    def sample(self,
               batch_size: int,
               reset_buffer: bool = False,
               random_samples: bool = True):
        indices = []
        if random_samples:
            indices = np.random.choice(len(self.buffer), batch_size, replace=True)
            replay_batch = [self.buffer[idx] for idx in indices]
        else:
            if batch_size == -1:
                replay_batch = [transition for transition in self.buffer]
            else:
                replay_batch = [self.buffer[-self.new_training_samples+idx] for idx in range(self.new_training_samples)]
        if reset_buffer:
            self.buffer.clear()
            self.state_deque = [deque(maxlen=self.n_steps+1) for x in range(self.agent_num)]
            self.action_deque = [deque(maxlen=self.n_steps+1) for x in range(self.agent_num)]
            self.reward_deque = [deque(maxlen=self.n_steps) for x in range(self.agent_num)]

        self.collected_trajectories = 0
        self.new_training_samples = 0

        copy_by_val_replay_batch = deepcopy(replay_batch)
        return copy_by_val_replay_batch, indices
