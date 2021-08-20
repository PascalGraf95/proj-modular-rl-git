import numpy as np
from .curriculum_strategy_blueprint import CurriculumStrategy


class RememberingCurriculum(CurriculumStrategy):
    Name = "RememberingCurriculum"

    def __init__(self):
        super().__init__()
        self.steps_since_remembering = 0
        self.remembering_frequency = 15
        self.temporary_task_level = 0

    def check_task_level_change_condition(self, average_reward, total_episodes_played, force=False):
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
            if self.steps_since_remembering >= self.remembering_frequency-1:
                temporary_task_level = np.random.randint(0, self.task_level)
                self.steps_since_remembering = -1
            self.steps_since_remembering += 1
        return temporary_task_level

    def get_logs(self):
        return self.temporary_task_level, self.average_episodes, self.average_reward





