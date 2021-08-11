import numpy as np
from .curriculum_strategy_blueprint import CurriculumStrategy


class LinearCurriculum(CurriculumStrategy):
    Name = "LinearCurriculum"

    def __init__(self):
        super().__init__()

    def check_task_level_change_condition(self, episode_reward_memory, episodes_played_memory, force=False):
        self.update_average_reward(episode_reward_memory)

        if episodes_played_memory - self.last_level_transition > self.average_episodes*3 or force:
            if self.task_level < self.number_of_tasks-1:
                if self.average_reward >= self.transition_value:
                    self.curriculum_sidechannel.send_current_task(task_level=self.task_level+1)
                    self.last_level_transition = episodes_played_memory
                    self.curriculum_sidechannel.unity_responded = False
                    self.unity_responded = False
        if not self.unity_responded:
            self.update_task_properties()
            return True
        return False





