import numpy as np
from .curriculum_strategy_blueprint import CurriculumStrategy
import ray


@ray.remote
class LinearCurriculum(CurriculumStrategy):
    Name = "LinearCurriculum"

    def __init__(self):
        super().__init__()

    def check_task_level_change_condition(self, average_reward, total_episodes_played, force=False):
        if total_episodes_played - self.last_level_transition > self.average_episodes*2 or force:
            if self.task_level < self.number_of_tasks-1:
                if average_reward >= self.transition_value or force:
                    self.task_level += 1
                    print("Transitioning to level ", self.task_level)
                    self.unity_responded = False
                    self.last_level_transition = total_episodes_played
                    return True
        return False
