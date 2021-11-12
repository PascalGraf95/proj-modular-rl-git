# Test the carla interface

# ---------------------------------------------------------------------------------------
# IMPORTS
# ---------------------------------------------------------------------------------------
import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla
import random
import time
import numpy as np
#import cv2
import math



# ---------------------------------------------------------------------------------------
# GLOBAL VARIABLES
# ---------------------------------------------------------------------------------------

# define global variables
SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480
SECONDS_PER_EPISODE = 10


# ---------------------------------------------------------------------------------------
# SIMULATION ENV
# ---------------------------------------------------------------------------------------
class CarEnv:
    
    # assign static variables
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None


    # ---------------------------------------------------------------------------------------
    # CONSTRUCTOR
    # ---------------------------------------------------------------------------------------
    def __init__(self):

        # connect to carla
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]

    # ---------------------------------------------------------------------------------------
    # RESET METHOD
    # ---------------------------------------------------------------------------------------
    def reset(self):
        # set up the actor and collision hist
        self.collision_hist = []
        self.actor_list = []

        # set transforms and vehicle
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
        self.actor_list.append(self.vehicle)

        # set the camera
        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.rgb_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.rgb_cam.set_attribute("fov", f"110")

        # Set up the camera
        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.sensor = self.world.spawn_actor(self.rgb_cam, transform, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        # Wait for a certain time until the simulation is ready
        # Send arbitrary command to vehicle
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(4)

        # Set up the colision sensor
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        # Wait until cam is running
        while self.front_camera is None:
            time.sleep(0.01)

        # check the episode start time
        self.episode_start = time.time()

        # Send arbitrary command to vehicle
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))

        # Return the front camera image
        return self.front_camera


    # ---------------------------------------------------------------------------------------
    # REGISTER COLLISION
    # ---------------------------------------------------------------------------------------
    def collision_data(self, event):
        self.collision_hist.append(event)

    # ---------------------------------------------------------------------------------------
    # PROCESS IMAGES
    # ---------------------------------------------------------------------------------------
    def process_img(self, image):
        i = np.array(image.raw_data)
        #np.save("iout.npy", i)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        i3 = i2[:, :, :3]
        if self.SHOW_CAM:
            cv2.imshow("",i3)
            cv2.waitKey(1)
        self.front_camera = i3

    
    # ---------------------------------------------------------------------------------------
    # STEP
    # ---------------------------------------------------------------------------------------
    def step(self, action):
        
        # Perform the action itself
        if action == 0:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0))
        if action == 1:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=-1*self.STEER_AMT))
        if action == 2:
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=1*self.STEER_AMT))

        # Get the vehicles velocity
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        # Determine the reward 
        if len(self.collision_hist) != 0:
            done = True
            reward = -200
            
        elif kmh < 50:
            done = False
            reward = -1

        else:
            done = False
            reward = 1

        # check if the maximum episode time is reached
        if self.episode_start + SECONDS_PER_EPISODE < time.time():
            done = True

        # return the observation, reward, done 
        return self.front_camera, reward, done, None
