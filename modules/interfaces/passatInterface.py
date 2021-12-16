#!/usr/bin/env python

#from _typeshed import Self
import numpy as np
#from mlagents_envs.environment import UnityEnvironment, ActionTuple
#from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
import sys
sys.path.append("/home/ai-admin/proj-modular-reinforcement-learning/modules/interfaces/")
from environments.passatEnvironment import PassatEnvironment
import cv2
import time


class Steps:
    def __init__(self, observation, reward):
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

class PassatInterface:

    observation = 0
    reward = 0
    done = False
    info = 0
    action_space = ""

    @staticmethod
    def connect():
        env = PassatEnvironment()
        PassatInterface.reset(env)
        return env

    @staticmethod
    def get_behavior_name(env: PassatEnvironment):
        return None

    @staticmethod
    def get_observation_shapes(env: PassatEnvironment):
        return [env.observation_space.shape]

    @staticmethod
    def get_random_action(env: PassatEnvironment, agent_number):
        return env.action_space.sample()

    @staticmethod
    def get_action_shape(env: PassatEnvironment):
        action_type_form = env.ACTION_TYPE_FORM
        action_type = env.action_space
        if action_type_form == "DISCRETE":
            return (action_type.n, )
        else:
            return action_type.shape[0]

    @staticmethod
    def get_action_type(env: PassatEnvironment):
        action_type_form = env.ACTION_TYPE_FORM
        if action_type_form == "DISCRETE":
            PassatInterface.action_space = "DISCRETE"
            return "DISCRETE"
        else:
            PassatInterface.action_space = "CONTINUOUS"
            return "CONTINUOUS"

    @staticmethod
    def get_agent_number(env: PassatEnvironment, behavior_name: str):
        return 1

    @staticmethod
    def get_steps(env: PassatEnvironment, behavior_name: str):
        if PassatInterface.done:
            decision_steps = Steps(None, None)
            terminal_steps = Steps(PassatInterface.observation, PassatInterface.reward)
        else:
            decision_steps = Steps(PassatInterface.observation, PassatInterface.reward)
            terminal_steps = Steps(None, None)

        return decision_steps, terminal_steps

    @staticmethod
    def reset(env: PassatEnvironment):
        PassatInterface.observation = env.reset()
        PassatInterface.reward = 0
        PassatInterface.done = False

    @staticmethod
    def step_action(env: PassatEnvironment, behavior_name: str, actions):
        
        if len(actions) == 0:
            actions = np.array([0.0])
        
        
        if PassatInterface.action_space == "DISCRETE":
            PassatInterface.observation, PassatInterface.reward, \
                PassatInterface.done, PassatInterface.info = env.step(actions[0][0])
        else:
            PassatInterface.observation, PassatInterface.reward, \
                PassatInterface.done, PassatInterface.info = env.step(actions[0])

    @staticmethod
    def get_acceleration(env: PassatEnvironment, behavior_name: str):
        return env.acceleration



# main loop
if __name__ == "__main__":

    # Create the environment
    env = PassatInterface.connect()
    
    # outer loop to run different episodes
    while True:
        
        # reset the environment
        PassatInterface.reset(env)

        # run the loop
        action = 0.0
        while True:

            # take a step in simulation
            observation, reward, done, _ = env.step(action)

            # check if the episode is over
            if done: break
