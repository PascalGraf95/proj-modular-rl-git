from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage
import uuid


class GameResultsSideChannel(SideChannel):
    """
    Side-Channel to communicate game results to determine ratings with Airhockey environments.
    """
    def __init__(self) -> None:
        super().__init__(uuid.UUID("2f487771-440f-4ffc-afd9-486650eb5b7b"))
        self.game_results = None

    # implement this method to receive messages from Unity
    def on_message_received(self, msg: IncomingMessage) -> None:
        self.game_results = msg.read_float32_list()
        # convert to int
        self.game_results = [int(x) for x in self.game_results]
        print(f"Received game result:  Agent score {self.game_results[0]}, Human score {self.game_results[1]}")

    # return current game result and reset it
    def get_game_result(self):
        if self.game_results is None:
            return None
        else:
            result = self.game_results
            self.game_results = None
            return result

