import numpy as np

from .meta_learning_blueprint import MetaLearning
from ..misc.replay_buffer import FIFOBuffer
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.optimizers import Adam, SGD
from ..misc.network_constructor import construct_network
import tensorflow as tf
from ..training_algorithms.agent_blueprint import Learner
import itertools
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Concatenate
import time
from collections import deque
from tensorflow.keras.utils import plot_model

class MetaController(MetaLearning):
    Name = "MetaController"

    ParameterSpace = {
        "WindowSize": int,
        "Alpha": float
    }

    def __init__(self, action_shape, observation_shapes, action_space, meta_learning_parameters, idx):
        self.action_space = action_space
        self.action_shape = action_shape
        self.observation_shapes = observation_shapes

        self.index = idx
        self.device = '/cpu:0'

        # region - Epsilon-Greedy Parameters -
        self.epsilon = meta_learning_parameters["Epsilon"]
        self.epsilon_decay = meta_learning_parameters["EpsilonDecay"]
        self.epsilon_min = meta_learning_parameters["EpsilonMin"]
        self.step_down = meta_learning_parameters["StepDown"]
        self.training_step = 0
        # endregion

        # region - UCB-Bandit Parameters -
        # Number of past episodes to store within UCB's global buffer
        self.window_size = meta_learning_parameters["UCBWindowSize"]
        # The global buffer contains the episode buffer's episodes
        self.global_buffer = deque(maxlen=self.window_size)
        # Max. length represents the max. number of steps within a single episode
        self.episode_buffer = deque(maxlen=1000)
        # Number of arms/exploration policies the bandit can choose from
        self.num_arms = meta_learning_parameters["ExplorationPolicies"]
        self.arm_play_count = np.zeros(self.num_arms)
        self.empirical_mean = np.zeros(self.num_arms)
        # Beta defines the weighting of the usage amount of arms in the process of choosing a new arm.
        self.beta = meta_learning_parameters["UCBBeta"]
        # The amount of specific arm usages can be count either '+1' per episode or alternatively per timestep.
        self.episodic_counting = meta_learning_parameters["EpisodicCounting"]
        self.count = 0
        self.arm_index = None
        self.most_chosen_arm = 0
        # endregion

    def act(self):
        """
        Pull one of the UCB-Bandit arms:
        In the context of Agent57 each arm represents one exploration policy pair (beta, gamma)
        Returns:
            arm_index (float): index of chosen arm j
        """
        # Every arm needs to be played once at least (for each actor)
        if self.count < self.num_arms:
            self.arm_index = self.count
            self.most_chosen_arm = self.arm_index
            self.count += 1
        else:
            if np.random.rand() < self.epsilon:
                self.arm_index = np.random.randint(self.num_arms)
            else:
                arm_play_count = np.zeros(self.num_arms)
                empirical_mean = np.zeros(self.num_arms)
                # Sum-up the rewards and determine the amount of episodes or timesteps the respective arm has been used
                # in. episode[:,0] contains the arm indices each timestep, episode[:,1] contains the extr. rewards each
                # timestep. episode[0, 0] gives the respective arm index.
                for episode in self.global_buffer:
                    episode = np.array(episode)
                    empirical_mean[episode[0, 0]] += np.sum(episode[:, 1])
                    if self.episodic_counting:
                        # Counting per episode
                        arm_play_count[episode[0, 0]] += 1
                    else:
                        # Alternative: Counting per timestep
                        arm_play_count[episode[0, 0]] += len(episode)
                empirical_mean = empirical_mean / (arm_play_count + 1e-6)
                self.arm_index = np.argmax(empirical_mean + self.beta * np.sqrt(1 / (arm_play_count + 1e-6)))

                # Log most frequently chosen arm within respective bandit window
                self.most_chosen_arm = np.argmax(arm_play_count)

        return self.arm_index

    def learning_step(self, arm_index, extrinsic_reward):
        """
        Push new arm configurations and rewards to episodic buffer.
        """
        self.episode_buffer.append([arm_index, extrinsic_reward])
        # Epsilon-Greedy Learning Step
        self.training_step += 1
        if self.epsilon >= self.epsilon_min and not self.step_down:
            self.epsilon *= self.epsilon_decay
        if self.training_step >= self.step_down and self.step_down:
            self.epsilon = self.epsilon_min

    def reset(self):
        """
        Push episode steps to global buffer as a single episode and empty episode buffer.
        """
        self.global_buffer.append(list(self.episode_buffer))
        self.episode_buffer.clear()
        return

    def get_logs(self):
        return {"MetaLearning/Agent{:02d}_BanditArm".format(self.index): self.most_chosen_arm}

    def prevent_checkpoint(self):
        return False

    @staticmethod
    def get_config(config_dict=None):
        if not config_dict:
            config_dict = MetaController.__dict__
        config_dict = {key: val for (key, val) in config_dict.items()
                       if not key.startswith('__')
                       and not callable(val)
                       and not type(val) is staticmethod
                       }
        return config_dict