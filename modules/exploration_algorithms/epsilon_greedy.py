import numpy as np
from .exploration_algorithm_blueprint import ExplorationAlgorithm


class EpsilonGreedy(ExplorationAlgorithm):
    """
    Epsilon-Greedy exploration algorithm acting randomly in epsilon percent of the cases.
    Epsilon will either decay over time or step down after a given number of episodes to the
    minimum value.
    """
    Name = "EpsilonGreedy"
    ActionAltering = True
    IntrinsicReward = False

    def __init__(self, action_shape, state_shape, action_space, parameters, trainer_configuration, idx):
        self.action_shape = action_shape
        self.action_space = action_space
        self.epsilon = parameters["Epsilon"]*parameters["ExplorationDegree"][idx]["scaling"]
        self.epsilon_decay = parameters["EpsilonDecay"]
        self.epsilon_min = parameters["EpsilonMin"]
        self.step_down = parameters["StepDown"]
        self.training_step = 0
        self.index = idx

    @staticmethod
    def get_config():
        config_dict = EpsilonGreedy.__dict__
        return ExplorationAlgorithm.get_config(config_dict)

    def act(self, decision_steps, terminal_steps):
        if len(decision_steps.agent_id):
            if np.random.rand() <= self.epsilon:
                if self.action_space == "DISCRETE":
                    return np.random.randint(0, self.action_shape, (len(decision_steps.agent_id), 1))
                else:
                    return np.random.uniform(-1.0, 1.0, (len(decision_steps.agent_id), self.action_shape))
        return None

    def boost_exploration(self):
        self.epsilon += 0.2

    def get_logs(self):
        return {"Exploration/Agent{:03d}Epsilon".format(self.index): self.epsilon}

    def get_intrinsic_reward(self, replay_batch):
        return replay_batch

    def prevent_checkpoint(self):
        if self.epsilon > self.epsilon_min * 10:
            return True
        return False

    def learning_step(self, replay_batch):
        self.training_step += 1
        if self.epsilon >= self.epsilon_min and not self.step_down:
            self.epsilon *= self.epsilon_decay

        if self.training_step >= self.step_down and self.step_down:
            self.epsilon = self.epsilon_min
