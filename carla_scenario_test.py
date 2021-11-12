import carla
import random
import time
import numpy as np
import math
import cv2
import json



# define global variables
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 40


# ---------------------------------------------------------------------------------------
# SIMULATION ENV
# ---------------------------------------------------------------------------------------
class CarlaEnv:
    
    # assign static variables
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    IM_WIDTH = IM_WIDTH
    IM_HEIGHT = IM_HEIGHT
    MPS_TO_KPH = 3.6
    KPH_TO_MPS = 1 / 3.6
    DISTANCE_BUMPER_COMP = 4.8

    # assign image
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

        # Choose a vehicle blueprint at random
        self.vehicle = random.choice(self.blueprint_library.filter('model3'))
        self.target =  random.choice(self.blueprint_library.filter('model3'))

    # ---------------------------------------------------------------------------------------
    # RESET METHOD
    # ---------------------------------------------------------------------------------------
    def reset(self, scenario_name):
        
        # destroy all actors
        for actor in self.actor_list:
            actor.destroy()
        self.actor_list = []

        # read a scenario
        self.scenario_name = scenario_name
        self.scenario_definition = self.read_scenario(self.scenario_name)

        # set transforms and vehicle
        # spawn the actor
        transform_target = carla.Transform(carla.Location(x=20 + self.scenario_definition["init_dx_target"], y=-20.0, z=2), carla.Rotation(yaw=0))
        transform_vehicle = carla.Transform(carla.Location(x=20, y=-20.0, z=2), carla.Rotation(yaw=0))
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

        # Wait for a certain time until the simulation is ready
        time.sleep(4)

        # Wait until cam is running
        while self.img_front_camera is None:
            time.sleep(0.01)

        # apply initial speeds
        self.actor_target.set_target_velocity(carla.Vector3D(self.scenario_definition['init_speed_target'] * self.KPH_TO_MPS, 0 , 0))
        self.actor_vehicle.set_target_velocity(carla.Vector3D(self.scenario_definition['init_speed_ego'] * self.KPH_TO_MPS, 0 , 0))

        # check the episode start time
        self.episode_start = time.time()

        # Return the front camera image
        return self.img_front_camera

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

        else:
            raise(NotImplementedError)

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
        # 1. APPLY SCENARIO TO ENVIRONMENT
        # *****************************************
        
        # 1.1 Manipulate target speed according to scenario
        # ===================================================
        # loop over all sequences in the target behaviour 
        for sequence in self.scenario_definition['target_behaviour_sequence']:
            
            # check which sequence is active
            if (time.time() - self.episode_start) < sequence["end_time"] and \
                (time.time() - self.episode_start) > sequence["start_time"]:

                # Print the current ID
                #print(round(time.time() - self.episode_start, 3), "in sequence ID: ", sequence["ID"])

                # A active sequemce has been found
                sequence_active = True
                target_speed = self.determine_target_speed_ramp(sequence, time.time() - self.episode_start)
                
                # apply the control
                self.actor_target.set_target_velocity(carla.Vector3D(target_speed * self.KPH_TO_MPS, 0 , 0))


        # 1.1 Manipulate target speed according to scenario
        # ===================================================
        # loop over all sequences in the target behaviour 
        for sequence in self.scenario_definition['speed_restriction_sequence']:

            # check which sequence is active
            if (time.time() - self.episode_start) > sequence["start_time"]:

                # Set the speed restriction
                self.speed_restriction = sequence["speed_restriction"]


        # *****************************************
        # 2. APPLY ACTION TO AGENT
        # *****************************************
        #self.actor_vehicle.set_target_velocity(carla.Vector3D(30 * self.KPH_TO_MPS, 0 , 0))



        # *****************************************
        # 3. CALCULATE REWARD
        # *****************************************
        reward = 0

        ## Initially, the 

        # 3.1 Reward based on the current speedlimit
        # ==========================================
        if self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH > self.speed_restriction:
            reward_speed_restriction = -50
        else:
            reward_speed_restriction = 0

        
        # 3.2 Reward based on the current headway
        # ==========================================
        
        # Calculate the headway
        x_vehicle = self.actor_vehicle.get_location().x
        x_target = self.actor_target.get_location().x
        ego_speed = self.actor_vehicle.get_velocity().x
        if ego_speed > 0.1:
            headway = (x_target - x_vehicle - self.DISTANCE_BUMPER_COMP) / ego_speed
        else:
            headway = 99

        # apply suggested reward function
        if headway < 0.5:
            reward_headway = -100
        elif headway < 1.75:
            reward_headway = -5
        elif headway < 1.9:
            reward_headway = 5
        elif headway < 2.1:
            reward_headway = 50
        elif headway < 2.25:
            reward_headway = 5
        elif headway < 10:
            reward_headway = 1
        elif headway > 10:
            reward_headway = 0.5


        # 3.2 Reward based on the current setspeed
        # ==========================================
        if abs(self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH - self.speed_set) < 1:
            reward_set_speed = 50

        elif abs(self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH - self.speed_set) < 2:
            reward_set_speed = 25

        elif abs(self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH - self.speed_set) < 3:
            reward_set_speed = 10
        else:
            reward_set_speed = -50


        # 3.4 Aggregate Rewards
        # ==========================================

        # check if target is far --> set speed matters
        if headway > 10:
            reward_motion = reward_set_speed

        # when the target is close, differenciate
        else:

            # if the target is in OK range and set speed is met, use it
            if reward_set_speed > 0 and headway > 1.9:
                reward_motion = reward_set_speed

            else:
                reward_motion = reward_headway

        # aggregate the final reward
        reward = reward_speed_restriction + reward_motion

        # *****************************************
        # 4. CHECK MAX TIMER
        # *****************************************
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        # *****************************************
        # 5. DEBUG
        # *****************************************
        img = self.img_front_camera
        #img = cv2.putText(self.img_front_camera, 'OpenCV', (10, 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)

        print("=================================================================")
        print(round(time.time() - self.episode_start, 3))
        print("TARGET SPEED      [kph]: ", self.actor_target.get_velocity().x * self.MPS_TO_KPH)
        print("AGENT  SPEED      [kph]: ", self.actor_vehicle.get_velocity().x * self.MPS_TO_KPH)
        print("SPEED RESTRICTION [kph]: ", self.speed_restriction)
        print("SPEED SETTING     [kph]: ", self.speed_set)
        print("HEADWAY           [ s ]: ", headway)
        print("REWARD                 : ", reward)


        # return the observation, reward, done 
        return img, reward, done, None


# main loop
if __name__ == "__main__":

    # Create the environment
    env = CarlaEnv()

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



    


