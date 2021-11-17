import numpy as np
import ray


class CurriculumStrategy:
    Name = "CurriculumStrategy"

    def __init__(self):
        self.last_level_transition = 0
        self.average_reward = 0

        self.task_level = 0
        self.number_of_tasks = 0
        self.transition_value = 0
        self.average_episodes = 0
        self.unity_responded = False
        self.level_transition = False

    def has_unity_responded(self):
        return self.unity_responded

    def check_task_level_change_condition(self, average_reward, total_episodes_played, force=False):
        return False

    def get_new_task_level(self, target_task_level):
        if not target_task_level:
            return None
        return self.task_level

    def get_average_episodes(self):
        return self.average_episodes

    def get_level_transition(self):
        return self.level_transition

    def update_task_properties(self, unity_responded, task_properties):
        if unity_responded:
            self.number_of_tasks = task_properties[0]
            self.task_level = task_properties[1]
            self.average_episodes = task_properties[2]
            self.transition_value = task_properties[3]
            self.unity_responded = unity_responded
            print("Task: {}/{}, Average Episodes {}, Average Reward {}".format(self.task_level+1, self.number_of_tasks,
                                                                               self.average_episodes,
                                                                               self.transition_value))

    def return_task_properties(self):
        return self.number_of_tasks, self.task_level, self.average_episodes, self.transition_value


@ray.remote
class NoCurriculumStrategy(CurriculumStrategy):
    def __init__(self):
        super().__init__()


class CurriculumCommunicator:
    def __init__(self, side_channel):
        self.side_channel = side_channel

    def get_task_properties(self):
        number_of_tasks = int(self.side_channel.task_info[0])
        task_level = int(self.side_channel.task_info[1])
        average_episodes = int(self.side_channel.task_info[2])
        transition_value = self.side_channel.task_info[3]
        unity_responded = self.side_channel.unity_responded
        return unity_responded, (number_of_tasks, task_level, average_episodes, transition_value)

    def set_task_number(self, task_level):
        self.side_channel.send_current_task(task_level)


