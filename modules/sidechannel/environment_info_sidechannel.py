from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage
import uuid
import re


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
            # split at ',' and ' ' and convert to list
            item_list = re.split(",|\s", self.environment_information_string)
            item_list = item_list[:-1]
            self.environment_information = {}
            for i in range(len(item_list)):
                if i % 2 == 0:
                    if item_list[i] in self.environment_information:
                        if not isinstance(self.environment_information[item_list[i]], list):
                            self.environment_information[item_list[i]] = [self.environment_information[item_list[i]]]
                        self.environment_information[item_list[i]].append(item_list[i + 1])
                    else:                        
                        self.environment_information[item_list[i]] = item_list[i + 1]
                else:
                    continue
            return self.environment_information
        return None
        
    