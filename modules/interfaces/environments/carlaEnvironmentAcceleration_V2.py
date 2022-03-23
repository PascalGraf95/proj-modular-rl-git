import carla
import random
import time
import numpy as np
import math
import cv2
import json
import os



# ---------------------------------------------------------------------------------------
# SIMULATION ENV
# ---------------------------------------------------------------------------------------
class CarlaEnvironment:
    
    # assign static variables
    SHOW_CAM = False
    IM_WIDTH = 640
    IM_HEIGHT = 480
    MPS_TO_KPH = 3.6
    KPH_TO_MPS = 1 / 3.6
    DISTANCE_BUMPER_COMP = 4.8
    SECONDS_PER_EPISODE = 30
    ACTION_TYPE_FORM = "CONTINUOUS"

    # Up to V14 (including 14)
    #MAX_ACCEL = 2.5 #m/s²
    #MAX_DECEL = 4.5 #m/s²

    # from V15 on
    MAX_ACCEL = 2.5 #m/s²
    MAX_DECEL = 2.5 #m/s²

    action_space = np.array([1])
    #observation_space = np.array([1, 2, 3, 4, 5]) # V1 - V7
    #observation_space = np.array([1, 2, 3, 4]) # V8, V11
    #observation_space = np.array([1, 2, 3, 4, 5]) # V9, V10
    #observation_space = np.array(["controlspeed_kph", "egospeed_kph", "headway"]) # V12 - V14
    observation_space = np.array(["controlspeed_kph", "egospeed_kph", "dx_m", "vx_rel_kph"]) # V15
    observation_space = np.array(["controlspeed_kph", "egospeed_kph", "dx_m", "vx_rel_kph", "previous_requ_acc"]) # V16
    DELTA_T = 0.100 #s

    # measurement variables
    MEASUREMENT_DICT = {"scenarioname": "", "timestamps": [], "setspeed": [], "speedrestriction": [], "targetspeed": [], "egospeed": [], "headway": [], "acc_requested":[], "acc_measured":[], "acc_calc": []}
    MEASUREMENT_ACTIVE = True
    SW_VERSION = "V16-TRY2"
    OUTPUT_PATH = "/home/ai-admin/Measurements/V2/"

    # Scenario variables
    SCENARIO_PATH = "/home/ai-admin/proj-modular-reinforcement-learning/scenarios/V2/scenarios/"

    # Assign image
    img_front_camera = None


    # ---------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------------------------------------------------
    def __init__(self):

        # connect to carla
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.world = self.client.load_world('Town06')
        self.blueprint_library = self.world.get_blueprint_library()

        # set up a sctor list
        self.actor_list = []

        # check the episode start time
        self.episode_start = time.time()

        # Choose a vehicle blueprint at random
        self.vehicle = random.choice(self.blueprint_library.filter('model3'))
        self.target =  random.choice(self.blueprint_library.filter('model3'))

        # Get all scenarios
        self.scenario_list_cat1 = os.listdir(self.SCENARIO_PATH + "/CAT1/")
        self.scenario_list_cat2 = os.listdir(self.SCENARIO_PATH + "/CAT2/")
        self.scenario_list_cat3 = os.listdir(self.SCENARIO_PATH + "/CAT3/")
        self.scenario_list_cat4 = os.listdir(self.SCENARIO_PATH + "/CAT4/")

    # ---------------------------------------------------------------------------------------
    # RESET METHOD
    # ---------------------------------------------------------------------------------------
    def reset(self):
        
        # destroy all actors
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

        # Load a scenario definition
        # Select a category first
        # - CAT4 two times in list to balance 50:50 between controlspeed scenarios and target follow scenarios)
        # - CAT1: 50:50 (controlspeed:follow)
        # - CAT2: follow
        # - CAT3: follow
        # - CAT4: controlspeed
        category = random.choice(["CAT1", "CAT2", "CAT3", "CAT4", "CAT4"])

        # Select a scenario name of the category
        if category == "CAT1":
            self.scenario_name = random.choice(self.scenario_list_cat1)
        elif category == "CAT2":
            self.scenario_name = random.choice(self.scenario_list_cat2)
        elif category == "CAT3":
            self.scenario_name = random.choice(self.scenario_list_cat3)
        elif category == "CAT4":
            self.scenario_name = random.choice(self.scenario_list_cat4)
        
        # Optionally override
        #self.scenario_name = "31_10.json"

        # Read the scenario
        self.scenario_definition = self.read_scenario(self.SCENARIO_PATH + category + "/" + self.scenario_name)

        # set transforms and vehicle
        # spawn the actor
        transform_target = carla.Transform(carla.Location(x=-250 + self.scenario_definition["init_dx_target"], y=-20.0, z=2), carla.Rotation(yaw=0))
        transform_vehicle = carla.Transform(carla.Location(x=-250, y=-20.0, z=2), carla.Rotation(yaw=0))
        self.speed_restriction = self.scenario_definition["init_speed_restriction"]
        self.speed_set = self.scenario_definition["init_speed_set"]
        self.actor_vehicle = self.world.spawn_actor(self.vehicle, transform_vehicle)
        self.actor_target = self.world.spawn_actor(self.target, transform_target)
        self.actor_list.append(self.actor_vehicle)
        self.actor_list.append(self.actor_target)

        # set the camera
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.IM_WIDTH}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.IM_HEIGHT}")
        self.rgb_cam.set_attribute("fov", f"110")

        # Set up the camera
        transform = carla.Transform(carla.Location(x=0.0, z=1.5))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.actor_vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        # Reset the measurement
        if len(self.MEASUREMENT_DICT["timestamps"]) > 0:
            self.save_measurement(self.MEASUREMENT_DICT["scenarioname"][:-5] + "_measurement_" + self.SW_VERSION + ".json")
        self.MEASUREMENT_DICT = {"scenarioname": "", "timestamps": [], "setspeed": [], "speedrestriction": [], "targetspeed": [], "egospeed": [], "headway": [], "acc_requested":[], "acc_measured":[], "acc_calc": []}

        # Wait for a certain time until the simulation is ready
        time.sleep(3)
        self.actor_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0))

        # apply initial speeds
        self.actor_target.set_target_velocity(carla.Vector3D(self.scenario_definition['init_speed_target'] * self.KPH_TO_MPS, 0 , 0))
        # since switch to accel interface, init speed zero is bad
        if self.scenario_definition['init_speed_ego'] == 0:
            self.actor_vehicle.set_target_velocity(carla.Vector3D(0.5, 0 , 0))
        else:
            self.actor_vehicle.set_target_velocity(carla.Vector3D(self.scenario_definition['init_speed_ego'] * self.KPH_TO_MPS, 0 , 0))
        self.target_speed = self.scenario_definition['init_speed_target']
        time.sleep(1)

        # Wait until cam is running
        while self.img_front_camera is None:
            time.sleep(0.01)

        # check the episode start time
        self.episode_start = time.time()
        self.lastrun = time.time()

        # Return the observation
        dx_rel = self.scenario_definition['init_dx_target'] - self.DISTANCE_BUMPER_COMP
        vx_rel = (self.scenario_definition['init_speed_ego'] - self.scenario_definition['init_speed_ego']) * self.MPS_TO_KPH
        self.previous_action_acc = 0.0

        # V16
        controlspeed = min(self.speed_set, self.speed_restriction)
        return np.array([controlspeed,\
             self.scenario_definition['init_speed_ego'] * self.MPS_TO_KPH, dx_rel, vx_rel, self.previous_action_acc])

        # V8, V11, V15: MIN
        controlspeed = min(self.speed_set, self.speed_restriction)
        return np.array([controlspeed,\
             self.scenario_definition['init_speed_ego'] * self.MPS_TO_KPH, dx_rel, vx_rel])

        # V12: MIN, headway
        controlspeed = min(self.speed_set, self.speed_restriction)
        return np.array([controlspeed,\
             self.scenario_definition['init_speed_ego'] * self.MPS_TO_KPH, 99])

        # V9, v10: back to 5 inputs
        #return np.array([self.speed_set, self.speed_restriction, self.scenario_definition['init_speed_ego'] * self.MPS_TO_KPH, dx_rel, vx_rel])

    # ---------------------------------------------------------------------------------------
    # PROCESS IMAGES
    # ---------------------------------------------------------------------------------------
    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.IM_HEIGHT, self.IM_WIDTH, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.img_front_camera = i3

    # ---------------------------------------------------------------------------------------
    # READ SCENARIO
    # ---------------------------------------------------------------------------------------
    def read_scenario(self, path):
 
        # Opening JSON file
        f = open(path,)
        
        # returns JSON object as a dictionary
        data = json.load(f)

        # return the read data
        return data

    # ---------------------------------------------------------------------------------------
    # SAVE MEASUREMENT
    # ---------------------------------------------------------------------------------------
    def save_measurement(self, name):
        
        # Dump the measurement dict as a json file
        with open(self.OUTPUT_PATH + name, 'w') as f:
            json.dump(self.MEASUREMENT_DICT, f)

    
    # ---------------------------------------------------------------------------------------
    # DETERMINE SPEED RAMP
    # ---------------------------------------------------------------------------------------
    def determine_target_speed_ramp(self, sequence, time_in_episode):

        # Check the sequence type
        if sequence["type"] == "RAMP":

            # check how much time has elapsed in the sequence
            time_in_sequence = time_in_episode - sequence["start_time"]

            # calculate the slope
            slope = (sequence["end_speed"] - sequence["start_speed"]) / (sequence["end_time"] - sequence["start_time"])
            
            # calculate the current target speed
            target_speed = slope * time_in_sequence + sequence["start_speed"]

        elif sequence["type"] == "RAMP_AX":
            
            # check how much time has elapsed in the sequence
            time_in_sequence = time_in_episode - sequence["start_time"]

            # calculate the slope
            slope = sequence["acceleration"]
            
            # calculate the current target speed in m/s
            target_speed_ms = slope * time_in_sequence + sequence["start_speed"] * self.KPH_TO_MPS

            # calculate the current target speed in kph
            target_speed = target_speed_ms * self.MPS_TO_KPH


        else:
            raise(NotImplementedError)

        # Bound the output
        if slope < 0:
            if target_speed < sequence["end_speed"]:
                target_speed = sequence["end_speed"]
        else:
            if target_speed > sequence["end_speed"]:
                target_speed = sequence["end_speed"]

        # return the target_speed
        return target_speed


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
            # apply the control
            speed = self.actor_vehicle.get_velocity().x
            action_acc = float(action) * self.MAX_DECEL * self.DELTA_T
            speed_set = speed + float(action) * self.MAX_DECEL * self.DELTA_T
            self.actor_vehicle.set_target_velocity(carla.Vector3D(speed_set, 0 , 0))
            print("braking")

        elif action > 0:
            # apply the control
            speed = self.actor_vehicle.get_velocity().x
            action_acc = float(action) * self.MAX_ACCEL * self.DELTA_T
            speed_set = speed + float(action) * self.MAX_ACCEL * self.DELTA_T
            self.actor_vehicle.set_target_velocity(carla.Vector3D(speed_set, 0 , 0))
            print("accelerating")

        elif action == 0:
            action_acc = 0.0
            print("nothing")


        # *****************************************
        # 2. SIMULATE FOR CERTAIN TIME
        # *****************************************
        """
        runtime = time.time()-self.lastrun
        if runtime < self.DELTA_T:
            time.sleep(self.DELTA_T - runtime)
        self.lastrun = time.time()
        """
        time.sleep(self.DELTA_T)


        # *****************************************
        # 3. GET THE OBSERVATIONS
        # *****************************************
        a_vehicle = self.actor_vehicle.get_acceleration().x
        a_vehicle_calc = (self.actor_vehicle.get_velocity().x- speed) / self.DELTA_T
        a_target = self.actor_target.get_acceleration().x
        x_vehicle = self.actor_vehicle.get_location().x
        x_target = self.actor_target.get_location().x
        v_vehicle = self.actor_vehicle.get_velocity().x
        v_target = self.actor_target.get_velocity().x
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

        """
        # V4 - V15 Reward

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
        """

        # V16 Reward

        # Definitions
        max_headway_to_be_rewarded = 5

        # Check if the vehicle is still driving
        if v_vehicle > 1:
            
            # Determine Headway
            headway = (x_target - x_vehicle - self.DISTANCE_BUMPER_COMP) / v_vehicle

            # apply suggested reward function
            if headway < 0.5:
                reward_headway = -1000
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

        else:
            # Set headway by assuming 1m/s, effectively turning it into stopping distance
            headway = headway = (x_target - x_vehicle - self.DISTANCE_BUMPER_COMP) / 1.0

            # Check if the target is also standing
            if abs(v_target) < 1:
                
                # apply suggested reward function
                if headway < 1.5:
                    reward_headway = -1000
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

            else:
                # Target is not in standstill
                reward_headway = -25
        

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

        """
        # V5 - V8 REWARD

        # Defintions
        max_reward = 50
        min_reward = -50
        max_reward_outside_target_range = 0
        min_reward_inside_target_range = 10
        target_range_max_deviation = 3
        outside_range_max_deviation = 130

        # Determine the deviation
        deviation_from_control_speed = abs(v_vehicle * self.MPS_TO_KPH - control_speed)
        
        # Check if the reward is in the target zone
        if deviation_from_control_speed <= 3:
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
        
        elif deviation_from_control_speed > 3:

            # Reward in target zone
            # Reward structure:
            #
            #  deviation 
            #   130                           3     0
            # | 0 ----------------------------|-----|
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
            reward_speed = min_reward

        """

        """
        #V9 Reward

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
        if deviation_from_control_speed <= 3:
            
            # Reward in target zone
            # Reward structure:
            #
            # | 50                      -----
            # |                        |
            # | 25               -------
            # |                  |
            # | 10         -------
            # |            |
            # | 0 ---------|-----|-----|-----|
            #  deviation   3     2     1     0

            # Check if the ego speed is in range
            if abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 1:
                reward_speed = 50

            elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 2:
                reward_speed = 25

            elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 3:
                reward_speed = 10
        
        elif deviation_from_control_speed > 3:

            # Reward in target zone
            # Reward structure:
            #
            #  deviation 
            #   130                           3     0
            # | 0 ----------------------------|-----|
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

            # Offset the reward based on if the acceleration points to the correct way
            if v_vehicle * self.MPS_TO_KPH < control_speed:
                if a_vehicle > 1.0:
                    reward_speed += 10
                else:
                    # Do nothing
                    pass

            elif v_vehicle * self.MPS_TO_KPH > control_speed:
                if a_vehicle < -1.0:
                    reward_speed += 10
                else:
                    # Do nothing
                    pass

        else:
            reward_speed = min_reward

        """

        """
        # V10 - V14

        # Defintions
        max_reward = 50
        min_reward = 0
        max_reward_outside_target_range = 10
        min_reward_inside_target_range = 10
        target_range_max_deviation = 3
        outside_range_max_deviation = 130

        # Determine the deviation
        deviation_from_control_speed = abs(v_vehicle * self.MPS_TO_KPH - control_speed)
        
        # Check if the reward is in the target zone
        if deviation_from_control_speed <= 3:
            
            # Reward in target zone
            # Reward structure:
            #
            # | 50                      -----
            # |                        |
            # | 25               -------
            # |                  |
            # | 10         -------
            # |            |
            # | 0 ---------|-----|-----|-----|
            #  deviation   3     2     1     0

            # Check if the ego speed is in range
            if abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 1:
                reward_speed = 50

            elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 2:
                reward_speed = 40

            elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 3:
                reward_speed = 30
        
        elif deviation_from_control_speed > 3 and v_vehicle * self.MPS_TO_KPH > 0:

            # Reward in target zone
            # Reward structure:
            #
            #  deviation 
            #   130                           3     0
            # | 0 ----------------------------|-----|
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

            # Offset the reward based on if the acceleration points to the correct way
            if v_vehicle * self.MPS_TO_KPH < control_speed:
                if a_vehicle > 1.0:
                    reward_speed += 10
                else:
                    # Do nothing
                    pass

            elif v_vehicle * self.MPS_TO_KPH > control_speed:
                if a_vehicle < -1.0:
                    reward_speed += 10
                else:
                    # Do nothing
                    pass
        
        # since accel interface: penalty for reversing
        elif v_vehicle * self.MPS_TO_KPH < 0:
            reward_speed = min_reward

        else:
            reward_speed = min_reward
        """

        """
        #V15 Reward (back to V9 reward)

        # Defintions
        max_reward = 50
        min_reward = -50
        max_reward_outside_target_range = -10
        min_reward_inside_target_range = 10
        target_range_max_deviation = 3
        outside_range_max_deviation = 130

        # Determine the deviation
        deviation_from_control_speed = abs(v_vehicle * self.MPS_TO_KPH - control_speed)
        
        # Check if the reward is in the target zone
        if deviation_from_control_speed <= 3:
            
            # Reward in target zone
            # Reward structure:
            #
            # | 50                      -----
            # |                        |
            # | 25               -------
            # |                  |
            # | 10         -------
            # |            |
            # | 0 ---------|-----|-----|-----|
            #  deviation   3     2     1     0

            # Check if the ego speed is in range
            if abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 1:
                reward_speed = 50

            elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 2:
                reward_speed = 25

            elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 3:
                reward_speed = 10
        
        elif deviation_from_control_speed > 3:

            # Reward in target zone
            # Reward structure:
            #
            #  deviation 
            #   130                           3     0
            # | 0 ----------------------------|-----|
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

            # Offset the reward based on if the acceleration points to the correct way
            if v_vehicle * self.MPS_TO_KPH < control_speed:
                if a_vehicle > 0.2:
                    reward_speed += 10
                else:
                    # Do nothing
                    pass

            elif v_vehicle * self.MPS_TO_KPH > control_speed:
                if a_vehicle < -0.2:
                    reward_speed += 10
                else:
                    # Do nothing
                    pass

        else:
            reward_speed = min_reward

        """

        #V16 Reward

        # Defintions
        max_reward = 50
        min_reward = -50
        max_reward_outside_target_range = -10
        min_reward_inside_target_range = 10
        target_range_max_deviation = 3
        outside_range_max_deviation = 130

        # Determine the deviation
        deviation_from_control_speed = abs(v_vehicle * self.MPS_TO_KPH - control_speed)
        
        # Check if the reward is in the target zone
        if deviation_from_control_speed <= 3:
            
            # Reward in target zone
            # Reward structure:
            #
            # | 50                      -----
            # |                        |
            # | 25               -------
            # |                  |
            # | 10         -------
            # |            |
            # | 0 ---------|-----|-----|-----|
            #  deviation   3     2     1     0

            # Check if the ego speed is in range
            if abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 1:
                reward_speed = 50

            elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 2:
                reward_speed = 25

            elif abs(v_vehicle * self.MPS_TO_KPH - control_speed) < 3:
                reward_speed = 10
        
        elif deviation_from_control_speed > 3:

            # Reward in target zone
            # Reward structure:
            #
            #  deviation 
            #   130                           3     0
            # | 0 ----------------------------|-----|
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

            # Offset the reward based on if the acceleration points to the correct way
            if v_vehicle * self.MPS_TO_KPH < control_speed:
                if a_vehicle_calc > 0.2:
                    reward_speed += 10
                else:
                    # Do nothing
                    pass

            elif v_vehicle * self.MPS_TO_KPH > control_speed:
                if a_vehicle_calc < -0.2:
                    reward_speed += 10
                else:
                    # Do nothing
                    pass

        else:
            reward_speed = min_reward


        # 4.3 Reward based on JERK (from V16 on)
        # ==========================================
        max_jerk = self.MAX_ACCEL + self.MAX_DECEL
        jerk = abs(action_acc - self.previous_action_acc)
        self.previous_action_acc = action_acc
        reward_jerk = -50 * jerk/max_jerk


        # 4.3 Aggregate Rewards
        # ==========================================

        # check if target is far --> set speed matters
        if headway > max_headway_to_be_rewarded:
            reward_motion = reward_speed

        # when the target is close, differenciate
        else:

            # if the target is in OK range and set speed is met, use it
            # V13: determine OK range differently, border 1
            # V14: border 3
            if deviation_from_control_speed < 3 and headway > 1.9:
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
        # from V16: add reward jerk (which is negative)
        reward = reward_motion + reward_jerk

        # normalize reward
        reward /= 50

        # FROM V5 on: Round the reward
        reward = round(reward, 2)


        # *****************************************
        # 5. APPLY SCENARIO TO ENVIRONMENT
        # *****************************************
        
        # 5.1 Manipulate target speed according to scenario
        # ===================================================
        # loop over all sequences in the target behaviour 
        for sequence in self.scenario_definition['target_behaviour_sequence']:

            # determine the endtime of the sequence
            if sequence["type"] == "RAMP":
                endtime = sequence["end_time"]

            elif sequence["type"] == "RAMP_AX":
                endtime = abs(((sequence["end_speed"] - sequence["start_speed"]) * self.KPH_TO_MPS) / sequence["acceleration"]) + sequence["start_time"]
            
            # check which sequence is active
            if (time.time() - self.episode_start) < endtime and \
                (time.time() - self.episode_start) > sequence["start_time"]:

                # A active sequence has been found
                sequence_active = True
                self.target_speed = self.determine_target_speed_ramp(sequence, time.time() - self.episode_start)
                
        # apply the control
        self.actor_target.set_target_velocity(carla.Vector3D(self.target_speed * self.KPH_TO_MPS, 0 , 0))
    

        # 5.2 Manipulate speed restriction according to scenario
        # ===================================================
        # loop over all sequences in the target behaviour 
        for sequence in self.scenario_definition['speed_restriction_sequence']:

            # check which sequence is active
            if (time.time() - self.episode_start) > sequence["start_time"]:

                # Set the speed restriction
                self.speed_restriction = sequence["speed_restriction"]


        # 5.3 Manipulate set speed according to scenario
        # ===================================================
        # loop over all sequences in the target behaviour 
        for sequence in self.scenario_definition['speed_set_sequence']:

            # check which sequence is active
            if (time.time() - self.episode_start) > sequence["start_time"]:

                # Set the speed set
                self.speed_set = sequence["speed_set"]
        

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
            self.MEASUREMENT_DICT["acc_measured"].append(a_vehicle)
            self.MEASUREMENT_DICT["acc_calc"].append(a_vehicle_calc)
            if action < 0:
                acc_requested = float(action) * self.MAX_DECEL
            else:
                acc_requested = float(action) * self.MAX_ACCEL
            self.MEASUREMENT_DICT["acc_requested"].append(acc_requested)


        # *****************************************
        # 7. DEBUG
        # *****************************************
        img = self.img_front_camera
        cv2.imshow("", img)
        cv2.waitKey(1)
        #img = cv2.putText(self.img_front_camera, 'OpenCV', (10, 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

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
        # V16
        controlspeed = min(self.speed_set, self.speed_restriction)
        return np.array([controlspeed, self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH, dx_rel, vx_rel, self.previous_action_acc]), reward, done, None

        # V8, V11, V15: MIN
        controlspeed = min(self.speed_set, self.speed_restriction)
        return np.array([controlspeed, self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH, dx_rel, vx_rel]), reward, done, None

        # V12 - 14: MIN, Headway
        #controlspeed = min(self.speed_set, self.speed_restriction)
        #return np.array([controlspeed, self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH, headway]), reward, done, None

        # V9, V10: back to 5 inputs again
        #return np.array([self.speed_set, self.speed_restriction, self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH, dx_rel, vx_rel]), reward, done, None