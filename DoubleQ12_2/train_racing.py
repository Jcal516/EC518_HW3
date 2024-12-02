import glob
import os
import sys
import numpy as np
import random
from reward import reward_function
from deepq import learn

# Locate and append CARLA .egg file to the Python path
try:
    egg_path = glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0]
    sys.path.append(egg_path)
    print(f"Adding Carla path: {egg_path}")
except IndexError:
    print("Carla egg file not found.")
    sys.exit(1)  # Exit if CARLA is not found

import carla

class CarlaEnv:
    def __init__(self):
        """Initialize the Carla Environment."""
        self.seed()
        
        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(1000.0)

        # Init world and blueprint library
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()

        # Initialize vehicle, sensors, and state variables
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
        self.done = False
        self.image_data = None

        self.setup_vehicle()

    def seed(self, seed=None):
        """Set the random seed."""
        if seed is None:
            seed = np.random.randint(0, 10000)
        random.seed(seed)
        np.random.seed(seed)
        return seed

    def render(self):
        """Return the current camera image."""
        if self.image_data is not None:
            return self.image_data
        else:
            return np.zeros((240, 320, 3))  # Black placeholder image

    def find_valid_spawn_point(self, vehicle_bp):
        """Find a valid spawn point that does not collide with existing actors."""
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for spawn_point in spawn_points:
            try:
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                if vehicle:
                    vehicle.destroy()
                    return spawn_point
            except:
                continue
        raise RuntimeError("No valid spawn point found.")

    def setup_vehicle(self):
        """Set up the vehicle in the CARLA world."""
        vehicle_bp = random.choice(self.blueprint_library.filter('vehicle'))
        if vehicle_bp.has_attribute('color'):
            color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
            vehicle_bp.set_attribute('color', color)

        spawn_point = self.find_valid_spawn_point(vehicle_bp)
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

        self.camera = self.create_camera(self.vehicle)

        collision_bp = self.blueprint_library.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self._on_collision)

        lane_invasion_bp = self.blueprint_library.find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(self._on_lane_invasion)

        self._update_next_waypoint()

    def _on_collision(self, event):
        self.collision_occurred = True

    def _on_lane_invasion(self, event):
        self.lane_invasion_occurred = True

    def _update_next_waypoint(self):
        self.current_position = self.vehicle.get_transform().location
        self.next_waypoint = self.world.get_map().get_waypoint(self.current_position).transform.location

    def create_camera(self, vehicle):
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute("image_size_x", str(320))
        camera_bp.set_attribute("image_size_y", str(240))
        camera_bp.set_attribute("fov", "110")
        transform = carla.Transform(carla.Location(x=1.5, z=1.0))
        camera = self.world.spawn_actor(camera_bp, transform, attach_to=vehicle)

        def image_callback(image):
            self.image_data = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))[:, :, :3]

        camera.listen(image_callback)
        return camera

    def reset(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()

        self.setup_vehicle()
        self.done = False
        return self.get_observation()

    def step(self, action):
        control = carla.VehicleControl(steer=action[0], throttle=action[1], brake=action[2])
        self.vehicle.apply_control(control)

        self._update_next_waypoint()
        current_distance = np.linalg.norm([
            self.current_position.x - self.next_waypoint.x,
            self.current_position.y - self.next_waypoint.y
        ])
        if self.previous_distance is None:
            self.previous_distance = current_distance

        self.world.tick()

        velocity = self.vehicle.get_velocity()
        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5

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
        self.collision_occurred = False
        self.lane_invasion_occurred = False

        return self.get_observation(), reward, self.done, {}

    def get_observation(self):
        if self.image_data is None:
            return np.zeros((240, 320, 3))
        return self.image_data

    def is_done(self):
        return self.collision_occurred

    def close(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        if self.collision_sensor:
            self.collision_sensor.destroy()
        if self.lane_invasion_sensor:
            self.lane_invasion_sensor.destroy()

def main():
    env = CarlaEnv()
    learn(env=env)

if __name__ == '__main__':
    main()
