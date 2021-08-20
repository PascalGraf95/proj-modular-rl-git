import numpy as np


class CurriculumStrategy:
    Name = "CurriculumStrategy"

    def __init__(self):
        self.last_level_transition = 0
        self.average_reward = 0

        self.task_level = 0
        self.number_of_tasks = 0
        self.transition_value = 0
        self.average_episodes = 0
        self.unity_responded = True

        self.level_transition = False

    def check_task_level_change_condition(self, average_reward, total_episodes_played, force=False):
        return False

    def get_new_task_level(self, total_episodes_played):
        return self.task_level

    def update_task_properties(self, unity_responded, task_properties):
        if unity_responded:
            self.number_of_tasks = task_properties[0]
            self.task_level = task_properties[1]
            self.average_episodes = task_properties[2]
            self.transition_value = task_properties[3]
            self.unity_responded = self.unity_responded

    def return_task_properties(self):
        return self.number_of_tasks, self.task_level, self.average_episodes, self.transition_value


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


