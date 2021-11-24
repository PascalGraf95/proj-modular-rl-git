#!/usr/bin/env python

#from _typeshed import Self
import numpy as np
#from mlagents_envs.environment import UnityEnvironment, ActionTuple
#from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
import sys
sys.path.append("/home/ai-admin/proj-modular-reinforcement-learning/modules/interfaces/")
from environments.carlaEnvironment import CarlaEnvironment
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

class CarlaInterface:

    observation = 0
    reward = 0
    done = False
    info = 0
    action_space = ""

    @staticmethod
    def connect():
        env = CarlaEnvironment()
        CarlaInterface.reset(env)
        return env

    @staticmethod
    def get_behavior_name(env: CarlaEnvironment):
        return None

    @staticmethod
    def get_observation_shapes(env: CarlaEnvironment):
        return [env.observation_space.shape]

    @staticmethod
    def get_random_action(env: CarlaEnvironment, agent_number):
        return env.action_space.sample()

    @staticmethod
    def get_action_shape(env: CarlaEnvironment):
        action_type_form = env.ACTION_TYPE_FORM
        action_type = env.action_space
        if action_type_form == "DISCRETE":
            return (action_type.n, )
        else:
            return action_type.shape[0]

    @staticmethod
    def get_action_type(env: CarlaEnvironment):
        action_type_form = env.ACTION_TYPE_FORM
        if action_type_form == "DISCRETE":
            CarlaInterface.action_space = "DISCRETE"
            return "DISCRETE"
        else:
            CarlaInterface.action_space = "CONTINUOUS"
            return "CONTINUOUS"

    @staticmethod
    def get_agent_number(env: CarlaEnvironment, behavior_name: str):
        return 1

    @staticmethod
    def get_steps(env: CarlaEnvironment, behavior_name: str):
        if CarlaInterface.done:
            decision_steps = Steps(None, None)
            terminal_steps = Steps(CarlaInterface.observation, CarlaInterface.reward)
        else:
            decision_steps = Steps(CarlaInterface.observation, CarlaInterface.reward)
            terminal_steps = Steps(None, None)

        return decision_steps, terminal_steps

    @staticmethod
    def reset(env: CarlaEnvironment):
        CarlaInterface.observation = env.reset()
        CarlaInterface.reward = 0
        CarlaInterface.done = False

    @staticmethod
    def step_action(env: CarlaEnvironment, behavior_name: str, actions):

        if CarlaInterface.action_space == "DISCRETE":
            CarlaInterface.observation, CarlaInterface.reward, \
                CarlaInterface.done, CarlaInterface.info = env.step(actions[0][0])
        else:
            CarlaInterface.observation, CarlaInterface.reward, \
                CarlaInterface.done, CarlaInterface.info = env.step(actions[0])



# main loop
if __name__ == "__main__":

    # Create the environment
    env = CarlaInterface.connect()
    
    # outer loop to run different episodes
    while True:
        
        # reset the environment
        CarlaInterface.reset(env)

        # run the loop
        action = 0.0
        while True:

            # take a step in simulation
            observation, reward, done, _ = env.step(action)

            # check if the episode is over
            if done: break
