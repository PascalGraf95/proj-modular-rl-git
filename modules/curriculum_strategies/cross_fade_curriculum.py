import numpy as np
from .curriculum_strategy_blueprint import CurriculumStrategy


class CrossFadeCurriculum(CurriculumStrategy):
    Name = "CrossFadeCurriculum"

    def __init__(self):
        super().__init__()
        self.remembering_probability = 0.0
        self.transition_episodes = 800
        self.temporary_task_level = 0

    def check_task_level_change_condition(self, average_reward, total_episodes_played, force=False):
        if self.remembering_probability < 0.01 or force:
            if total_episodes_played - self.last_level_transition > self.average_episodes*2 or force:
                if self.task_level < self.number_of_tasks-1:
                    if self.average_reward >= self.transition_value or force:
                        self.level_transition = True
                        self.last_level_transition = total_episodes_played
                        self.unity_responded = False
                        self.task_level += 1
                        return True
        return False

    def get_new_task_level(self, total_episodes_played):
        temporary_task_level = self.task_level
        if self.task_level > 0 and self.unity_responded:
            if np.random.rand() <= self.remembering_probability:
                temporary_task_level = np.random.randint(0, self.task_level)
                self.remembering_probability = np.max([self.remembering_probability - 1/self.transition_episodes, 0])
        if self.remembering_probability <= 0.01:
            self.level_transition = False
        return temporary_task_level





