import glob
import os
import sys
import gym
import numpy as np
import random
from reward import reward_function
from deepq import learn

try:
    egg_path = glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(egg_path)
    print(f"Adding Carla path: {egg_path}")
except IndexError:
    print("Carla egg file not found.")
    sys.exit(1)  # Exit if Carla is not found

import carla

class CarlaEnv(gym.Env):
    def __init__(self):
        super(CarlaEnv, self).__init__()

        #Seed
        self.seed()
        
        # Connect to CARLA
        self.client = carla.Client('localhost', 4000)
        self.client.set_timeout(1000.0)

        # Init world
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        #init vehicle
        self.vehicle = None
        self.camera = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None

        self.collision_occurred = False
        self.lane_invasion_occurred = False

        self.current_position = None
        self.next_waypoint = None
        self.previous_distance = None

        self.target_speed = 30

        self.previous_distance = None

        self.setup_vehicle()

        #self.target_fps = target_fps
        self.clock = None
        self.done = False

    def seed(self, seed=None):
        """Set the seed for random number generation in the environment."""
        if seed is None:
            seed = np.random.randint(0, 10000)  # Generate a random seed if none is provided
            random.seed(seed)           # Set the seed for Pythons random module
            np.random.seed(seed)        # Set the seed for NumPys random number generator

        # If CARLA allows setting a random seed (e.g., for procedural generation), set it here
        # self.world.set_seed(seed)  # Uncomment if CARLA has a similar method (CARLA does not currently have this)
        
        return seed



    def render(self):
        # Return the current camera image as an array
        if self.image_data is not None:
            return self.image_data
        else:
            return np.zeros((240, 320, 3))  # If no image data is available yet, return a black image

        
    def find_valid_spawn_point(self,vehicle_bp):
        """Find a valid spawn point that does not collide with existing actors."""
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for spawn_point in spawn_points:
            try:
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                if self.vehicle is not None:  # Check if spawn was successful
                    self.vehicle.destroy()  # Remove the vehicle after check
                    return spawn_point
            except:
                print("Spawn failed")
                continue
        raise RuntimeError("No valid spawn point found.")

        
    def setup_vehicle(self):
            """Set up the vehicle in the CARLA world."""
            # Get a random blueprint for the vehicle
            vehicle_bp = random.choice(self.blueprint_library.filter('vehicle'))
            if vehicle_bp.has_attribute('color'):
                color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                vehicle_bp.set_attribute('color', color)


            # Spawn the vehicle at a random location
            spawn_points = self.find_valid_spawn_point(vehicle_bp)
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_points)

            # Set a fixed camera
            self.camera = self.create_camera(self.vehicle)

            # Set up sensors
            # Collision sensor
            collision_bp = self.blueprint_library.find('sensor.other.collision')
            self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
            self.collision_sensor.listen(self._on_collision)

            # Lane invasion sensor
            lane_invasion_bp = self.blueprint_library.find('sensor.other.lane_invasion')
            self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
            self.lane_invasion_sensor.listen(self._on_lane_invasion)

            # Set next waypoint for the agent
            self._update_next_waypoint()

    def _on_collision(self, event):
        """Callback when collision occurs."""
        self.collision_occurred = True

    def _on_lane_invasion(self, event):
        """Callback when lane invasion occurs."""
        self.lane_invasion_occurred = True

    def _update_next_waypoint(self):
        """Update the next waypoint for the agent to navigate to."""
        # Carla's waypoints API to get the next waypoint
        self.current_position = self.vehicle.get_transform().location
        self.next_waypoint = self.world.get_map().get_waypoint(self.current_position).transform.location

    def create_camera(self, vehicle):
            """Create a camera attached to the vehicle."""
            camera_bp = self.blueprint_library.find('sensor.camera.rgb')
            camera_bp.set_attribute("image_size_x", str(320))
            camera_bp.set_attribute("image_size_y", str(240))
            camera_bp.set_attribute("fov", "110")
            transform = carla.Transform(carla.Location(x=1.5, z=1.0))  # Adjust the position of the camera
            camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

            # Create an attribute to hold the latest image
            self.image_data = None

            # Define a callback to update the image
            def image_callback(image):
                self.image_data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]

            # Listen for new images using the callback
            camera.listen(image_callback)

            return camera
    
    def reset(self):
        """Reset the environment to an initial state."""
        # Destroy the current vehicle if it exists
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()


        # Setup the vehicle again
        self.setup_vehicle()
        self.done = False

        # Get the initial observation
        return self.get_observation()
    
    def step(self, action):
        """Take a step in the environment based on the action."""
        # Perform the action
        #print(action)
        control = carla.VehicleControl()
        control.steer = action[0]
        control.throttle = action[1]
        control.brake = action[2]
        self.vehicle.apply_control(control)
        
        self._update_next_waypoint()
        current_distance = np.linalg.norm(np.array([self.current_position.x, self.current_position.y]) - 
                                      np.array([self.next_waypoint.x, self.next_waypoint.y]))
        
        if self.previous_distance is None:
            self.previous_distance = current_distance
        # Wait for the simulation to step
        self.world.tick()


        #time.sleep(0.1)
        
        velocity = self.vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        # Calculate reward and done flag
        reward = reward_function(
                                collision=self.collision_occurred,
                                speed=speed,
                                lane_invasion=self.lane_invasion_occurred,
                                current_position=(self.current_position.x, self.current_position.y),
                                next_waypoint=(self.next_waypoint.x, self.next_waypoint.y),
                                target_speed=self.target_speed,
                                previous_distance=self.previous_distance
        )
        
        self.previous_distance = current_distance
        
        self.done = self.is_done()


        # Reset flags for collision and lane invasion for the next step
        self.collision_occurred = False
        self.lane_invasion_occurred = False

        # Get the new observation
        obs = self.get_observation()

        return obs, reward, self.done, {}

    def get_observation(self):
        """Get the current observation from the camera."""
        # Ensure the image has been captured
        if self.image_data is None:
            print("Waiting for camera data...")
            return np.zeros((240, 320, 3))  # Return a default black image as placeholder

        # Return the latest image
        return self.image_data


    def is_done(self):
        """Check if the episode is done."""
        # Implement your termination logic here
         # Episode is done if a collision occurred or if other conditions are met (e.g., going off track)
        if self.collision_occurred:
            return True
        return False

    def close(self):
        """Clean up the environment."""
        if self.vehicle is not None:
            self.vehicle.destroy()
        if self.camera is not None:
            self.camera.destroy()
        if self.collision_sensor is not None:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor is not None:
            self.lane_invasion_sensor.destroy()

#The default parameters for training a agent can be found in deepq.py
def main():

    """ 
    Train a Deep Q-Learning agent 
    """ 
    #Initialize your carla env above and train the Deep Q-Learning agent
    env = CarlaEnv()

    learn(env=env)
if __name__ == '__main__':
    main()

