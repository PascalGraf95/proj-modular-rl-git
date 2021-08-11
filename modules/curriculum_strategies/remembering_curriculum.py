import numpy as np
from .curriculum_strategy_blueprint import CurriculumStrategy


class RememberingCurriculum(CurriculumStrategy):
    Name = "RememberingCurriculum"

    def __init__(self):
        super().__init__()
        self.steps_since_remembering = 0
        self.remembering_frequency = 15
        self.temporary_task_level = 0

    def check_task_level_change_condition(self, episode_reward_memory, episodes_played_memory, force=False):
        if self.task_level > 0:
            self.temporary_task_level = self.task_level
            if self.steps_since_remembering == 0:
                self.curriculum_sidechannel.send_current_task(task_level=self.task_level)
            if self.steps_since_remembering >= self.remembering_frequency-1:
                self.temporary_task_level = np.random.randint(0, self.task_level)
                self.curriculum_sidechannel.send_current_task(task_level=self.temporary_task_level)
                self.steps_since_remembering = -1
            self.steps_since_remembering += 1

        self.update_average_reward(episode_reward_memory)
        if episodes_played_memory - self.last_level_transition > self.average_episodes*3 or force:
            if self.task_level < self.number_of_tasks-1:
                if self.average_reward >= self.transition_value or force:
                    self.curriculum_sidechannel.send_current_task(task_level=self.task_level+1)
                    self.last_level_transition = episodes_played_memory
                    self.unity_responded = False
                    self.curriculum_sidechannel.unity_responded = False
                    self.temporary_task_level = self.task_level
                    self.steps_since_remembering = 0
        if not self.unity_responded:
            self.update_task_properties()
            return True
        return False

    def get_logs(self):
        return self.temporary_task_level, self.average_episodes, self.average_reward





