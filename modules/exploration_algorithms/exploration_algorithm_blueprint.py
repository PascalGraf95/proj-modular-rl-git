import numpy as np
from modules.misc.replay_buffer import FIFOBuffer


class ExplorationAlgorithm:
    Name = "None"
    ActionAltering = False
    IntrinsicReward = False

    ParameterSpace = {}

    def __init__(self, action_shape, observation_shape, action_space, parameters, trainer_configuration, idx):
        self.action_space = action_space
        self.action_shape = action_shape

    @staticmethod
    def get_config(config_dict=None):
        if not config_dict:
            config_dict = ExplorationAlgorithm.__dict__
        config_dict = {key: val for (key, val) in config_dict.items()
                       if not key.startswith('__')
                       and not callable(val)
                       and not type(val) is staticmethod
                       }
        return config_dict

    def act(self, decision_steps, terminal_steps):
        return

    def boost_exploration(self):
        return

    def get_logs(self):
        return {}

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    def learning_step(self, replay_batch):
        return

    def prevent_checkpoint(self):
        return False

    def reset(self):
        return

    def epsilon_greedy(self, decision_steps):
        return None

    @staticmethod
    def calculate_intrinsic_reward(replay_buffer: FIFOBuffer):
        return True
