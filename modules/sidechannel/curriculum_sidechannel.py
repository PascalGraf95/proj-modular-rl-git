from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid


class CurriculumSideChannelTaskInfo(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f4"))
        self.task_info = [1, 0, 30, 10000]
        self.unity_responded = True

    # implement this method to receive messages from Unity
    def on_message_received(self, msg: IncomingMessage) -> None:
        self.task_info = msg.read_float32_list()
        self.unity_responded = True

    def send_current_task(self, task_level) -> None:
        current_task = [task_level, 1.0]
        # Create an outgoing message
        msg = OutgoingMessage()
        msg.write_float32_list(current_task)
        # method to queue the data we want to send
        super().queue_message_to_send(msg)

    def get_task_info(self):
        return self.task_info

