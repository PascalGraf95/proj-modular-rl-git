#!/usr/bin/env python

import numpy as np
import gym
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel

class Steps:
    def __init__(self, observation, reward, is_decision_step):
        if is_decision_step:
            if np.any(observation[0]):
                self.agent_id = np.reshape(np.array(0), 1)
                self.obs = [np.expand_dims(observation, axis=0)]
                self.reward = np.reshape(np.array(reward), 1)
            else:
                self.agent_id = np.empty(0)
                self.obs = [np.empty((0))]
                self.reward = np.empty(0)
        else:
            if np.any(observation):
                self.agent_id = np.reshape(np.array(0), 1)
                self.obs = [np.expand_dims(observation, axis=0)]
                self.reward = np.reshape(np.array(reward), 1)
            else:
                self.agent_id = np.empty(0)
                self.obs = [np.empty((0))]
                self.reward = np.empty(0)

    def __len__(self):
        return 1


class OpenAIGymInterface:
    """
    This interface translates the universal commands used in this repository for environment communication to
    interact with OpenAI Gym environments.

    SUPPORTED GYM-VERSION: v0.26.0 upwards
    """
    observation = 0
    reward = 0
    done = False
    truncated = False
    terminated = False
    info = 0
    action_space = ""

    @staticmethod
    def connect(environment_path):
        return gym.make(environment_path)

    @staticmethod
    def get_interface_name():
        return "OpenAIGym"

    @staticmethod
    def get_behavior_name(env: gym.Env):
        return None, None

    @staticmethod
    def get_observation_shapes(env: gym.Env):
        return [env.observation_space.shape]

    @staticmethod
    def get_random_action(env: gym.Env, agent_number):
        return env.action_space.sample()

    @staticmethod
    def get_action_shape(env: gym.Env, _placeholder):
        action_type = env.action_space
        if type(action_type) == gym.spaces.Discrete:
            return (action_type.n, )
        else:
            return action_type.shape[0]

    @staticmethod
    def get_action_type(env: gym.Env):
        action_type = env.action_space
        if type(action_type) == gym.spaces.Discrete:
            OpenAIGymInterface.action_space = "DISCRETE"
            return "DISCRETE"
        else:
            OpenAIGymInterface.action_space = "CONTINUOUS"
            return "CONTINUOUS"

    @staticmethod
    def get_agent_number(env: gym.Env, behavior_name: str):
        OpenAIGymInterface.reset(env)
        decision_steps, terminal_steps = OpenAIGymInterface.get_steps(env, behavior_name)
        return 1, 0

    @staticmethod
    def get_steps(env: gym.Env, behavior_name: str):
        if OpenAIGymInterface.done:
            decision_steps = Steps(None, None, False)
            terminal_steps = Steps(OpenAIGymInterface.observation, OpenAIGymInterface.reward, True)
        else:
            decision_steps = Steps(OpenAIGymInterface.observation, OpenAIGymInterface.reward, True)
            terminal_steps = Steps(None, None, False)

        return decision_steps, terminal_steps

    @staticmethod
    def reset(env: gym.Env):
        OpenAIGymInterface.observation, OpenAIGymInterface.info = env.reset()
        OpenAIGymInterface.reward, OpenAIGymInterface.done = 0, False
        OpenAIGymInterface.terminated, OpenAIGymInterface.truncated = False, False

    @staticmethod
    def step_action(env: gym.Env, action_type: str,
                    behavior_name: str,
                    actions,
                    behavior_clone_name=None,
                    actions_clone=None):
        if OpenAIGymInterface.action_space == "DISCRETE":
            OpenAIGymInterface.observation, OpenAIGymInterface.reward, \
                OpenAIGymInterface.terminated, OpenAIGymInterface.truncated, OpenAIGymInterface.info = env.step(actions[0][0])
        else:
            OpenAIGymInterface.observation, OpenAIGymInterface.reward, \
                OpenAIGymInterface.terminated, OpenAIGymInterface.truncated, OpenAIGymInterface.info = env.step(actions[0])

        OpenAIGymInterface.done = OpenAIGymInterface.terminated or OpenAIGymInterface.truncated

if __name__ == '__main__':
    env = FlattenObservation(gym.make('MountainCar-v0', render_mode='human'))
    OpenAIGymInterface.reset(env)
    print(OpenAIGymInterface.get_action_shape(env, None))
    print(OpenAIGymInterface.get_action_type(env))
    print(OpenAIGymInterface.get_observation_shapes(env))
    decision_step, terminal_step = OpenAIGymInterface.get_steps(env, None)
    while True:
        while not OpenAIGymInterface.done:
            env.render()
            actions = OpenAIGymInterface.get_random_action(env, 1)
            OpenAIGymInterface.step_action(env, "", "", [actions])
            decision_step, terminal_step = OpenAIGymInterface.get_steps(env, None)
        OpenAIGymInterface.reset(env)
    env.close()

