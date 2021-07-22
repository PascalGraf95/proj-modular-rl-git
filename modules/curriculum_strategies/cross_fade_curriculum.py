import numpy as np
from .curriculum_strategy_blueprint import CurriculumStrategy


class CrossFadeCurriculum(CurriculumStrategy):
    Name = "CrossFadeCurriculum"

    def __init__(self):
        super().__init__()
        self.remembering_probability = 0.0
        self.transition_episodes = 800
        self.temporary_task_level = 0

    def check_task_level_change_condition(self, episode_reward_memory, episodes_played_memory):
        if self.task_level > 0:
            self.temporary_task_level = self.task_level

            if np.random.rand() <= self.remembering_probability:
                self.temporary_task_level = np.random.randint(0, self.task_level)
                self.remembering_probability = np.max([self.remembering_probability - 1/self.transition_episodes, 0])
            self.curriculum_sidechannel.send_current_task(task_level=self.temporary_task_level)

        self.update_average_reward(episode_reward_memory)
        if self.remembering_probability < 0.01:
            if episodes_played_memory - self.last_level_transition > self.average_episodes*3:
                if self.task_level < self.number_of_tasks-1:
                    if self.average_reward >= self.transition_value:
                        self.curriculum_sidechannel.send_current_task(task_level=self.task_level+1)
                        self.last_level_transition = episodes_played_memory
                        self.curriculum_sidechannel.unity_responded = False
                        self.unity_responded = False
                        self.temporary_task_level = self.task_level
                        self.remembering_probability = 1.0
        if not self.unity_responded:
            self.update_task_properties()
            return True
        return False

    def get_logs(self):
        return self.temporary_task_level, self.average_episodes, self.average_reward





