import numpy as np


class CurriculumStrategy:
    Name = "CurriculumStrategy"

    def __init__(self):
        self.last_level_transition = 0
        self.curriculum_sidechannel = None
        self.average_reward = 0

        self.task_level = 0
        self.number_of_tasks = 0
        self.transition_value = 0
        self.average_episodes = 0
        self.unity_responded = True

    def register_side_channels(self, curriculum_sidechannel):
        self.curriculum_sidechannel = curriculum_sidechannel

    def update_average_reward(self, episode_reward_memory):
        if len(episode_reward_memory) > 0:
            self.average_reward = np.mean(list(episode_reward_memory)[-self.average_episodes:])

    def check_task_level_change_condition(self, episode_reward_memory, episodes_played_memory):
        self.update_average_reward(episode_reward_memory)
        return False

    def update_task_properties(self):
        self.number_of_tasks = int(self.curriculum_sidechannel.task_info[0])
        self.task_level = int(self.curriculum_sidechannel.task_info[1])
        self.average_episodes = int(self.curriculum_sidechannel.task_info[2])
        self.transition_value = self.curriculum_sidechannel.task_info[3]
        self.unity_responded = self.curriculum_sidechannel.unity_responded

    def get_logs(self):
        return self.task_level, self.average_episodes, self.average_reward
