#!/usr/bin/env python

import numpy as np
#from mlagents_envs.environment import UnityEnvironment, ActionTuple
#from mlagents_envs.side_channel.engine_configuration_channel import EngineConfig, EngineConfigurationChannel
from environments.carlaEnvironment import CarlaEnvironment
import cv2


class CarlaInterface:

    observation = 0
    reward = 0
    done = False
    info = 0
    action_space = ""

    @staticmethod
    def connect():
        return CarlaEnvironment()

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
        action_type = env.action_space
        if type(action_type) == gym.spaces.Discrete:
            return (action_type.n, )
        else:
            return action_type.shape[0]

    @staticmethod
    def get_action_type(env: CarlaEnvironment):
        action_type = env.action_space
        if type(action_type) == gym.spaces.Discrete:
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
        CarlaInterface.reward, CarlaInterface.done = 0, False

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

    # define the different scenarios
    scenarios = ["scenario_1.json", "scenario_2.json", "scenario_3.json"]
    
    # outer loop to run different episodes
    for scenario in scenarios:
        
        # reset the environment
        env.reset(scenario)

        # run the loop
        action = None
        while True:

            # take a step in simulation
            observation, reward, done, _ = env.step(action)

            # check if the episode is over
            if done: break

            # Show the camera
            cv2.imshow("", observation)
            cv2.waitKey(1)
