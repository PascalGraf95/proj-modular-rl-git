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


# ---------------------------------------------------------------------------------------
# CONNECT TO CARLA
# ---------------------------------------------------------------------------------------

# Register the client
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Request the world
world = client.get_world()

# Check for the available maps and load a new one
print(client.get_available_maps())
world = client.load_world('Town06')


# ---------------------------------------------------------------------------------------
# MANIPULATE THE WEATHER
# ---------------------------------------------------------------------------------------

weather = carla.WeatherParameters(
    cloudiness=10.0,
    precipitation=10.0,
    sun_altitude_angle=70.0)
world.set_weather(weather)


# ---------------------------------------------------------------------------------------
# SPAWN SOME ACTORS
# ---------------------------------------------------------------------------------------

# Acess the blueprint library
blueprint_library = world.get_blueprint_library()

# Choose a vehicle blueprint at random
vehicle = random.choice(blueprint_library.filter('model3'))
target =  random.choice(blueprint_library.filter('model3'))

# Modify the color of the target
target.set_attribute('color', '255,0,0')

# spawn the actor
transform_target = carla.Transform(carla.Location(x=120, y=-20, z=40), carla.Rotation(yaw=0))
transform_vehicle = carla.Transform(carla.Location(x=100, y=-20, z=40), carla.Rotation(yaw=0))
actor_vehicle = world.spawn_actor(vehicle, transform_vehicle)
actor_target = world.spawn_actor(target, transform_target)

print(actor_target.get_velocity())
actor_target.set_target_velocity(carla.Vector3D(50, 0 , 0))
actor_vehicle.set_target_velocity(carla.Vector3D(50, 0 , 0))