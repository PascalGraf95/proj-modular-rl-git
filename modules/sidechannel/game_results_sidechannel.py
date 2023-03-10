from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage
import uuid


class GameResultsSideChannel(SideChannel):
    """
    Side-Channel to communicate game results to determine player ratings with self-play environments.
    """
    def __init__(self) -> None:
        super().__init__(uuid.UUID("2f487771-440f-4ffc-afd9-486650eb5b7b"))
        self.game_results = None

    def on_message_received(self, msg: IncomingMessage) -> None:
        # Convert the incoming message consisting of two floats to a list of integers.
        self.game_results = msg.read_float32_list()
        self.game_results = [int(x) for x in self.game_results]

    def get_game_results(self):
        if self.game_results:
            tmp_game_results = self.game_results.copy()
            self.game_results = None
            return tmp_game_results
        return None











