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
        # convert incoming string in the form of "key1 value1,key2 value2,..." to a tuple
        if self.environment_information_string:
            # split add ',' and convert to list
            kvp_string_list = self.environment_information_string.split(",")
            # split the strings in the list into key and value and add to a tuple to have non exclusive keys
            self.environment_information = tuple(item.split(" ") for item in kvp_string_list)
            # remove empty last element of tuple
            self.environment_information = self.environment_information[:-1]
            return self.environment_information
        return None
        
    