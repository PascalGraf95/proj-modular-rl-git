#import carla
import random
import time
import numpy as np
import math
import cv2
import json
import os



# ---------------------------------------------------------------------------------------
# TESTING ENV
# ---------------------------------------------------------------------------------------
class PassatEnvironment:
    
    # assign static variables
    MPS_TO_KPH = 3.6
    KPH_TO_MPS = 1 / 3.6
    DISTANCE_BUMPER_COMP = 4.8
    SECONDS_PER_EPISODE = 30
    ACTION_TYPE_FORM = "CONTINUOUS"
    action_space = np.array([1])
    observation_space = np.array([1, 2, 3, 4, 5])
    DELTA_T = 0.100 #s

    # measurement variables
    MEASUREMENT_DICT = {"scenarioname": "", "timestamps": [], "setspeed": [], "speedrestriction": [], "targetspeed": [], "egospeed": [], "headway": []}
    OUTPUT_DIRECTORY = "/home/ai-admin/proj-modular-reinforcement-learning/training/summaries/211122_011414_RL_ACC_TEST_v2_normalized_reward/best/"
    MEASUREMENT_ACTIVE = False

    # Scenario variables
    SCENARIO_PATH = "/home/ai-admin/proj-modular-reinforcement-learning/scenarios/"

    # assign image
    img_front_camera = None


    # ---------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------------------------------------------------
    def __init__(self):

        # set up a sctor list
        self.actor_list = []

        # check the episode start time
        self.episode_start = time.time()

    # ---------------------------------------------------------------------------------------
    # RESET METHOD
    # ---------------------------------------------------------------------------------------
    def reset(self):
        
        # destroy all actors
        self.actor_list = []

        # set transforms and vehicle
        self.speed_restriction = 30
        self.speed_set = 20
        self.speed_ego_mps = 0

        # Reset the measurement
        if len(self.MEASUREMENT_DICT["timestamps"]) > 0:
            self.save_measurement(self.MEASUREMENT_DICT["scenarioname"] + "_measurement.json")
        self.MEASUREMENT_DICT = {"scenarioname": "", "timestamps": [], "setspeed": [], "speedrestriction": [], "targetspeed": [], "egospeed": [], "headway": []}

        # Wait for a certain time until the simulation is ready
        time.sleep(4)

        # check the episode start time
        self.episode_start = time.time()
        self.lastrun = time.time()

        # Return the observation
        dx_rel = 0
        vx_rel = 0
        return np.array([self.speed_set, self.speed_restriction, self.speed_ego_mps * self.MPS_TO_KPH, dx_rel, vx_rel])

    # ---------------------------------------------------------------------------------------
    # SAVE MEASUREMENT
    # ---------------------------------------------------------------------------------------
    def save_measurement(self, name):
        
        # Dump the measurement dict as a json file
        with open(name, 'w') as f:
            json.dump(self.MEASUREMENT_DICT, f)


    # ---------------------------------------------------------------------------------------
    # STEP
    # ---------------------------------------------------------------------------------------
    def step(self, action):
        
        # *****************************************
        # 0. INITIALIZE
        # *****************************************
        done = False


        # *****************************************
        # 1. APPLY ACTION TO AGENT
        # *****************************************
        if action < 0:
            self.acceleration = float(abs(action))
            print("braking")
        elif action > 0:
            self.acceleration = float(abs(action))
            print("accelerating")
        elif action == 0:
            print("nothing")


        # *****************************************
        # 2. SIMULATE FOR CERTAIN TIME
        # *****************************************
        runtime = time.time()-self.lastrun
        if runtime < self.DELTA_T:
            time.sleep(self.DELTA_T - runtime)
        self.lastrun = time.time()


        # *****************************************
        # 3. GET THE OBSERVATIONS
        # *****************************************
        x_vehicle = 0
        x_target = 10
        v_vehicle = 0
        v_target = 10
        a_vehicle = 0
        a_target = 0
        dx_rel = x_target - x_vehicle - self.DISTANCE_BUMPER_COMP
        vx_rel = v_target * self.MPS_TO_KPH - v_vehicle * self.MPS_TO_KPH


        # *****************************************
        # 4. CALCULATE REWARD
        # *****************************************
        reward = 0

        # 4.1 Reward based on the current headway
        # ==========================================

        """
        # V1 - V4 Reward

        # Determine Headway
        if v_vehicle > 0.1:
            headway = (x_target - x_vehicle - self.DISTANCE_BUMPER_COMP) / v_vehicle
        else:
            headway = 99

        # apply suggested reward function
        if headway < 0.5:
            reward_headway = -100
            #done = True
        elif headway < 1.75:
            reward_headway = -25
        elif headway < 1.9:
            reward_headway = 10
        elif headway < 2.1:
            reward_headway = 50
        elif headway < 2.25:
            reward_headway = 10
        elif headway < 10:
            reward_headway = 1
        elif headway > 10:
            reward_headway = 0.5
        """

        # V1 - V5 Reward

        # Definitions
        max_headway_to_be_rewarded = 5

        # Determine Headway
        if v_vehicle > 0.1:
            headway = (x_target - x_vehicle - self.DISTANCE_BUMPER_COMP) / v_vehicle
        else:
            headway = 99

        # apply suggested reward function
        if headway < 0.5:
            reward_headway = -100
            #done = True
        elif headway < 1.75:
            reward_headway = -25
        elif headway < 1.9:
            reward_headway = 10
        elif headway < 2.1:
            reward_headway = 50
        elif headway < 2.25:
            reward_headway = 10
        elif headway < max_headway_to_be_rewarded:
            reward_headway = 1
        elif headway > max_headway_to_be_rewarded:
            reward_headway = 0.5

        # 4.2 Reward based on the current control speed
        # ==========================================
        # Determine the control speed
        control_speed = min(self.speed_restriction, self.speed_set)

        """ 
        # V3 REWARD
        
        # Check if the ego speed is in range
        if abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 1:
            reward_speed = 50

        elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 2:
            reward_speed = 25

        elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 3:
            reward_speed = 10
        else:
            #Check if the tendency is correct
            if v_vehicle * self.MPS_TO_KPH < control_speed:
                if a_vehicle > 0:
                    reward_speed = -25
                else:
                    reward_speed = -50

            elif v_vehicle * self.MPS_TO_KPH > control_speed:
                if a_vehicle < 0:
                    reward_speed = -25
                else:
                    reward_speed = -50 
        """

        """"
        # V4 REWARD
        # Check if the ego speed is in range
        if abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 1:
            reward_speed = 50

        elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 2:
            reward_speed = 25

        elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 3:
            reward_speed = 10
        else:
            #Check if the tendency is correct
            if v_vehicle * self.MPS_TO_KPH < control_speed:
                if a_vehicle > 1.0:
                    reward_speed = -25
                else:
                    reward_speed = -50

            elif v_vehicle * self.MPS_TO_KPH > control_speed:
                if a_vehicle < -1.0:
                    reward_speed = -25
                else:
                    reward_speed = -50 
        """

        # V5 REWARD

        # Defintions
        max_reward = 50
        min_reward = -50
        max_reward_outside_target_range = -25
        min_reward_inside_target_range = 10
        target_range_max_deviation = 3
        outside_range_max_deviation = 130

        # Determine the deviation
        deviation_from_control_speed = abs(v_vehicle * self.MPS_TO_KPH - control_speed)
        
        # Check if the reward is in the target zone
        if deviation_from_control_speed > 3:

            # Reward in target zone
            # Reward structure:
            #
            #  deviation 
            #   130                           3     0
            # | 0 ----------------------------|-----|
            # |                               |
            # | -25                           |
            # |                             /
            # |                          /
            # |                       /
            # |                    /
            # |                 /
            # |              /
            # |           /
            # |        /
            # | -50 /

            # Determine the reward
            slope = (max_reward_outside_target_range - min_reward) / (outside_range_max_deviation - target_range_max_deviation)
            reward_speed = max_reward_outside_target_range - slope * (deviation_from_control_speed - target_range_max_deviation)

        else:
            # Reward in target zone
            # Reward structure:
            #
            # | 50             /
            # |               /
            # |              /
            # | 10          /
            # |            |
            # | 0 ---------|-----|
            #  deviation   3     0

            # Determine the reward
            slope = (max_reward - min_reward_inside_target_range) / (target_range_max_deviation - 0)
            reward_speed = max_reward - deviation_from_control_speed * slope
        


        # 4.3 Aggregate Rewards
        # ==========================================

        # check if target is far --> set speed matters
        if headway > max_headway_to_be_rewarded:
            reward_motion = reward_speed

        # when the target is close, differenciate
        else:

            # if the target is in OK range and set speed is met, use it
            if reward_speed > 0 and headway > 1.9:
                reward_motion = reward_speed

            # set speed is overshoot
            elif v_vehicle * self.MPS_TO_KPH > control_speed:
                reward_motion = reward_speed

            # Set speed is not overshot
            # Target is close
            # Use the headway reward
            else:
                reward_motion = reward_headway

        # aggregate the final reward
        reward = reward_motion

        # normalize reward
        reward /= 50

        # *****************************************
        # 6. SAVE MEASUREMENT
        # *****************************************
        if self.MEASUREMENT_ACTIVE:
            self.MEASUREMENT_DICT["scenarioname"] = self.scenario_name
            self.MEASUREMENT_DICT["timestamps"].append(time.time() - self.episode_start)
            self.MEASUREMENT_DICT["setspeed"].append(self.speed_set)
            self.MEASUREMENT_DICT["speedrestriction"].append(self.speed_restriction)
            self.MEASUREMENT_DICT["targetspeed"].append(v_target * self.MPS_TO_KPH)
            self.MEASUREMENT_DICT["egospeed"].append(v_vehicle * self.MPS_TO_KPH)
            self.MEASUREMENT_DICT["headway"].append(headway)


        # *****************************************
        # 7. DEBUG
        # *****************************************
        print("=================================================================")
        print(round(time.time() - self.episode_start, 3))
        print("TARGET SPEED      [kph]: ", v_target * self.MPS_TO_KPH)
        print("AGENT  SPEED      [kph]: ", v_vehicle * self.MPS_TO_KPH)
        print("SPEED RESTRICTION [kph]: ", self.speed_restriction)
        print("SPEED SETTING     [kph]: ", self.speed_set)
        print("HEADWAY           [ s ]: ", headway)
        print("REWARD                 : ", reward)


        # *****************************************
        # 8. CHECK ABORTIONS
        # *****************************************
        if self.episode_start + self.SECONDS_PER_EPISODE < time.time():
            done = True

        if dx_rel < 1:
            done = True

        # return the observation, reward, done 
        return np.array([self.speed_set, self.speed_restriction, v_vehicle, dx_rel, vx_rel]), reward, done, None