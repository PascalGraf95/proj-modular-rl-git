from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage
import uuid


class EnvironmentInfoSideChannel(SideChannel):
    """
    Side-Channel to communicate environment information like the reward composition.
    """
    def __init__(self) -> None:
        super().__init__(uuid.UUID("92744089-f2c0-49f9-ba9e-1968f1944e28"))
        self.environment_information_string = None
        self.environment_information = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        # Convert the incoming message consisting of two floats to a list of integers.
        self.environment_information_string = msg.read_string()

    def get_environment_information_from_string(self):
        # convert incoming string in the form of "key1 value1,key2 value2,..." to a dictionary
        if self.environment_information_string:
            self.environment_information = dict(item.split(" ") for item in self.environment_information.split(","))
            return self.environment_information
        return None
        
    