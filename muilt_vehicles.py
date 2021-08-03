import glob
import os
import sys
import json

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import argparse
import carla
import logging
import math
import pygame
import random
import queue
import threading
import concurrent.futures
import numpy as np
from bounding_box import create_kitti_datapoint
from constants import *
import image_converter
from dataexport import *

parser = argparse.ArgumentParser()
parser.add_argument("--overwrite", action="store_true", default=False, help="ow dataset. append otherwise")
parser.add_argument("--camera-fov", default="120", type=str, help="Camera FOV")
parser.add_argument("--ds-interval", default=60, type=int, help="Interval between frames to be exported")
args = parser.parse_args()


""" OUTPUT FOLDER GENERATION """
PHASE = "training"
OUTPUT_FOLDER = os.path.join("_out", PHASE)
folders = ['calib', 'image_2', 'label_2', 'velodyne', 'planes']


def maybe_create_dir(path):
    if not os.path.exists(directory):
        os.makedirs(directory)


for folder in folders:
    directory = os.path.join(OUTPUT_FOLDER, folder)
    maybe_create_dir(directory)

""" DATA SAVE PATHS """
GROUNDPLANE_PATH = os.path.join(OUTPUT_FOLDER, 'planes/{0:06}.txt')
LIDAR_PATH = os.path.join(OUTPUT_FOLDER, 'velodyne/{0:06}.bin')
LABEL_PATH = os.path.join(OUTPUT_FOLDER, 'label_2/{0:06}.txt')
IMAGE_PATH = os.path.join(OUTPUT_FOLDER, 'image_2/{0:06}.png')
CALIBRATION_PATH = os.path.join(OUTPUT_FOLDER, 'calib/{0:06}.txt')
STATS_FILE = os.path.join(OUTPUT_FOLDER, 'dslog.json')

class SynchronyModel(object):
    def __init__(self):
        self.world, self.init_setting, self.client, self.traffic_manager = self._make_setting()
        self.blueprint_library = self.world.get_blueprint_library()
        self.classes = ['Car', 'Pedestrian', 'Cyclist']
        self.non_player = []
        self.actor_list = []
        self.frame = None
        self.player = None
        self.captured_frame_no = self.current_captured_frame_num()
        self.sensors = []
        self._queues = []
        self.main_image = None
        self.depth_image = None
        self.point_cloud = None
        self.extrinsic = None
        self.intrinsic, self.my_camera = self._span_spectator(location=None, rotation=None)
        self.locations = self.create_locations()
        self.weather_conditions = self.create_weather_presets()
        self.ds_counter = self.init_ds_counter()
        self._span_non_player()

    def create_locations(self):
        locations = [
            carla.Transform(carla.Location(x=-55.640671, y=3.514651, z=5.624344), carla.Rotation(pitch=-30.182434, yaw=-169.737045, roll=0.000025)), # original
            carla.Transform(carla.Location(x=50.8, y=-13.9, z=5.12), carla.Rotation(pitch=-2.48, yaw=156.4, roll=0.000004)), 
            # carla.Transform(carla.Location(x=-55.640671, y=3.514651, z=5.624344), carla.Rotation(pitch=0, yaw=-169.737045, roll=0.000025)), # debug
            carla.Transform(carla.Location(x=173.431839, y=53.041073, z=4.272343), carla.Rotation(pitch=-38.675350, yaw=132.347687, roll=0.000067)),
            carla.Transform(carla.Location(x=160.032928, y=71.224144, z=5.341573), carla.Rotation(pitch=-41.482201, yaw=-140.674652, roll=0.000042)),
            carla.Transform(carla.Location(x=247.920151, y=73.704063, z=6.082335), carla.Rotation(pitch=-22.306694, yaw=-131.522003, roll=0.000102)), 
            carla.Transform(carla.Location(x=-57.657494, y=3.750844, z=5.813370), carla.Rotation(pitch=-26.488121, yaw=-20.079067, roll=0.000140)),
            carla.Transform(carla.Location(x=-67.881264, y=9.866474, z=6.102913), carla.Rotation(pitch=-21.914791, yaw=-103.430275, roll=0.000133)),
            ]
        return locations

    def create_weather_presets(self):
        weather_conditions = [

            # early morning with fog
            carla.WeatherParameters(cloudiness=20.000000, wind_intensity=0.350000, 
                                    precipitation=0.000000, precipitation_deposits=0.000000, wetness=0.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=177.0, 
                                    fog_density=30.000000, fog_distance=0.000000, fog_falloff=0.000000),

            # early morning
            carla.WeatherParameters(cloudiness=20.000000, wind_intensity=0.350000, 
                                    precipitation=0.000000, precipitation_deposits=0.000000, wetness=0.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=177.0, 
                                    fog_density=0.000000, fog_distance=0.000000, fog_falloff=0.000000),

            # sunny morning
            carla.WeatherParameters(cloudiness=0.000000, wind_intensity=0.350000, 
                                    precipitation=0.000000, precipitation_deposits=0.000000, wetness=0.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=130, 
                                    fog_density=0.000000, fog_distance=0.000000, fog_falloff=0.000000),

            # rainy morning
            carla.WeatherParameters(cloudiness=80.000000, wind_intensity=0.850000, 
                                    precipitation=80.000000, precipitation_deposits=50.000000, wetness=40.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=130, 
                                    fog_density=20.000000, fog_distance=0.000000, fog_falloff=0.000000),

            # sunny early afternoon
            carla.WeatherParameters(cloudiness=0.000000, wind_intensity=0.350000, 
                                    precipitation=0.000000, precipitation_deposits=0.000000, wetness=0.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=60, 
                                    fog_density=0.000000, fog_distance=0.000000, fog_falloff=0.000000),

            # foggy early afternoon
            carla.WeatherParameters(cloudiness=0.000000, wind_intensity=0.350000, 
                                    precipitation=0.000000, precipitation_deposits=0.000000, wetness=0.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=60, 
                                    fog_density=60.000000, fog_distance=0.000000, fog_falloff=0.000000),

            # rainy afternoon
            carla.WeatherParameters(cloudiness=20.000000, wind_intensity=0.350000,
                                    precipitation=0.000000, precipitation_deposits=50.000000, wetness=0.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=30.000000, 
                                    fog_density=30.000000, fog_distance=0.000000, fog_falloff=0.000000),

            # rainy evening with fog
            carla.WeatherParameters(cloudiness=20.000000, wind_intensity=0.350000,
                                    precipitation=5.000000, precipitation_deposits=50.000000, wetness=0.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=8.000000, 
                                    fog_density=30.000000, fog_distance=0.000000, fog_falloff=0.000000),
                        
            # clear night
            carla.WeatherParameters(cloudiness=20.000000, wind_intensity=0.350000,
                                    precipitation=0.000000, precipitation_deposits=0.000000, wetness=0.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=-7.000000, 
                                    fog_density=0.000000, fog_distance=0.000000, fog_falloff=0.000000),

            # rainy night
            carla.WeatherParameters(cloudiness=0.000000, wind_intensity=0.350000,
                                    precipitation=30.000000, precipitation_deposits=50.000000, wetness=20.000000,
                                    sun_azimuth_angle=0.000000, sun_altitude_angle=-7.000000, 
                                    fog_density=0.000000, fog_distance=0.000000, fog_falloff=0.000000),
            ]

        weather_conditions = [carla.WeatherParameters.WetSunset]

        return weather_conditions

    def set_weather(self, w):
        print(self.world.get_weather())
        weather = carla.WeatherParameters(sun_altitude_angle=70.0)
        self.world.set_weather(w)
        print(self.world.get_weather())

    def __enter__(self):
        # set the sensor listener function
        def make_queue(register_event):
            q = queue.Queue()
            register_event(q.put)
            self._queues.append(q)

        make_queue(self.world.on_tick)
        for sensor in self.sensors:
            make_queue(sensor.listen)
        return self

    def init_ds_counter(self):
        dsc = {}
        for w in self.weather_conditions:
            for loc in self.locations:  
                for c in self.classes:
                    if w not in dsc.keys():
                        dsc[w] = {}
                    if loc not in dsc[w].keys():
                        dsc[w][loc] = {}
                    dsc[w][loc][c] = 0
        return dsc

    def current_captured_frame_num(self):
        # Figures out which frame number we currently are on
        # This is run once, when we start the simulator in case we already have a dataset.
        # The user can then choose to overwrite or append to the dataset.
        label_path = os.path.join(OUTPUT_FOLDER, 'label_2/')
        num_existing_data_files = len(
            [name for name in os.listdir(label_path) if name.endswith('.txt')])
        print(num_existing_data_files)
        if num_existing_data_files == 0 or args.overwrite:
            return 0
        else:
            logging.info("Continuing recording data on frame number {}".format(
                num_existing_data_files))
            return num_existing_data_files

    def tick(self, timeout):
        # Drive the simulator to the next frame and get the data of the current frame
        self.frame = self.world.tick()
        data = [self._retrieve_data(q, timeout) for q in self._queues]
        assert all(x.frame == self.frame for x in data)
        return data

    def _retrieve_data(self, sensor_queue, timeout):
        while True:
            data = sensor_queue.get(timeout=timeout)
            if data.frame == self.frame:
                return data

    def __exit__(self, *args, **kwargs):
        # cover the world settings
        self.world.apply_settings(self.init_setting)

    def _make_setting(self):
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_global_distance_to_leading_vehicle(1.0)
        traffic_manager.set_synchronous_mode(True)
        # synchrony model and fixed time step
        init_setting = world.get_settings()
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 steps  per second
        world.apply_settings(settings)
        return world, init_setting, client, traffic_manager

    def _span_spectator(self, location=None, rotation=None):
        """create our target vehicle"""
        spectator = self.world.get_spectator()

        if location is None or rotation is None:
            location = spectator.get_transform().location
            rotation = spectator.get_transform().rotation

        spectator.set_transform(carla.Transform(location, rotation))

        k, my_camera = self._span_sensor(spectator)
        # self.actor_list.append(spectator) # we don't want for the spectator to be destroyed upon client disconnection
        self.player = spectator
        
        return k, my_camera

    def _span_sensor(self, player):
        """create camera, depth camera and lidar and attach to the target vehicle"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_d_bp = self.blueprint_library.find('sensor.camera.depth')
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')

        camera_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_bp.set_attribute('fov', args.camera_fov) # 45 60 90 120 150 180

        camera_d_bp.set_attribute('image_size_x', str(WINDOW_WIDTH))
        camera_d_bp.set_attribute('image_size_y', str(WINDOW_HEIGHT))
        camera_d_bp.set_attribute('fov', args.camera_fov) # 45 60 90 120 150 180

        lidar_bp.set_attribute('range', '200')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('upper_fov', '2')
        lidar_bp.set_attribute('lower_fov', '-26.8')
        lidar_bp.set_attribute('points_per_second', '320000')
        lidar_bp.set_attribute('channels', '32')

        transform_sensor = carla.Transform(carla.Location(x=0, y=0, z=0))   # we move the player (observer actor) instead. 

        my_camera = self.world.spawn_actor(camera_bp, transform_sensor, attach_to=player)
        my_camera_d = self.world.spawn_actor(camera_d_bp, transform_sensor, attach_to=player)
        my_lidar = self.world.spawn_actor(lidar_bp, transform_sensor, attach_to=player)

        self.actor_list.append(my_camera)
        self.actor_list.append(my_camera_d)
        self.actor_list.append(my_lidar)
        self.sensors.append(my_camera)
        self.sensors.append(my_camera_d)
        self.sensors.append(my_lidar)

        # camera intrinsic
        k = np.identity(3)
        k[0, 2] = WINDOW_WIDTH_HALF
        k[1, 2] = WINDOW_HEIGHT_HALF
        f = WINDOW_WIDTH / (2.0 * math.tan(float(args.camera_fov) * math.pi / 360.0))
        k[0, 0] = k[1, 1] = f
        return k, my_camera

    def _span_non_player(self):
        """create autonomous vehicles and people"""
        blueprints = self.world.get_blueprint_library().filter(FILTERV)
        # blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 2]   # spawn only bikes
        blueprints = [x for x in blueprints if not x.id.endswith('isetta')]
        blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
        blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
        blueprints = [x for x in blueprints if not x.id.endswith('t2')]
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)
        if NUM_OF_VEHICLES < number_of_spawn_points:
            random.shuffle(spawn_points)
            number_of_vehicles = NUM_OF_VEHICLES
        elif NUM_OF_VEHICLES > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, NUM_OF_VEHICLES, number_of_spawn_points)
            number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot
            batch.append(SpawnActor(blueprint, transform))

        vehicles_id = []
        for response in self.client.apply_batch_sync(batch):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_id.append(response.actor_id)
        vehicle_actors = self.world.get_actors(vehicles_id)
        self.non_player.extend(vehicle_actors)
        self.actor_list.extend(vehicle_actors)

        for i in vehicle_actors:
            i.set_autopilot(True, self.traffic_manager.get_port())

        blueprintsWalkers = self.world.get_blueprint_library().filter(FILTERW)
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(NUM_OF_WALKERS):
            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = self.client.apply_batch_sync(batch, True)
        walker_speed2 = []
        walkers_list = []
        all_id = []
        walkers_id = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = self.client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put altogether the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = self.world.get_actors(all_id)

        for i in range(len(walkers_list)):
            walkers_id.append(walkers_list[i]["id"])
        walker_actors = self.world.get_actors(walkers_id)
        self.non_player.extend(walker_actors)
        self.actor_list.extend(all_actors)
        self.world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        self.world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(self.world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))

        print('spawned %d walkers and %d vehicles, press Ctrl+C to exit.' % (len(walkers_id), len(vehicles_id)))

    def _save_training_files(self, datapoints, point_cloud):
        """ Save data in Kitti dataset format """
        logging.info("Attempting to save at frame no {}, frame no: {}".format(self.frame, self.captured_frame_no))
        groundplane_fname = GROUNDPLANE_PATH.format(self.captured_frame_no)
        lidar_fname = LIDAR_PATH.format(self.captured_frame_no)

        kitti_fname = LABEL_PATH.format(self.captured_frame_no)
        img_fname = IMAGE_PATH.format(self.captured_frame_no)
        calib_filename = CALIBRATION_PATH.format(self.captured_frame_no)

        save_groundplanes(groundplane_fname, self.player, LIDAR_HEIGHT_POS)
        save_ref_files(OUTPUT_FOLDER, self.captured_frame_no)

        save_image_data(img_fname, self.main_image)
        save_kitti_data(kitti_fname, datapoints)
        save_calibration_matrices(calib_filename, self.intrinsic, self.extrinsic)

        save_lidar_data(lidar_fname, point_cloud)

    def generate_datapoints(self, image):
        """ Returns a list of datapoints (labels and such) that are generated this frame together with the main image
        image """
        datapoints = []
        image = image.copy()
        # Remove this
        rotRP = np.identity(3)

        if not GEN_DATA:
            return image, datapoints

        # Calculate depth map once for the current frame (instead of once for each agent)
        depth_map = image_converter.depth_to_array(self.depth_image)

        # Stores all datapoints for the current frame
        for agent in self.non_player:
            image, kitti_datapoint = create_kitti_datapoint(
                agent, self.intrinsic, self.extrinsic, image, depth_map, self.player, rotRP)
            if kitti_datapoint:
                datapoints.append(kitti_datapoint)

        return image, datapoints


def draw_image(surface, image, blend=False):
    # array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    # array = np.reshape(array, (image.height, image.width, 4))
    # array = array[:, :, :3]
    # array = array[:, :, ::-1]
    array = image[:, :, ::-1]
    image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))


def should_quit(sync_mode):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:
                print("scroll wheel up")
                sync_mode.world.set_weather(carla.WeatherParameters(sun_altitude_angle = sync_mode.world.get_weather().sun_altitude_angle + 1))
                print("sun_altitude_angle=", sync_mode.world.get_weather().sun_altitude_angle, sep='')
            elif event.button == 5:
                print("scroll wheel down")
                sync_mode.world.set_weather(carla.WeatherParameters(sun_altitude_angle = sync_mode.world.get_weather().sun_altitude_angle - 1))
                print("sun_altitude_angle=", sync_mode.world.get_weather().sun_altitude_angle, sep='')

    return False

def get_font():
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, 14)


def main():
    pygame.init()
    display = pygame.display.set_mode(
        (WINDOW_WIDTH, WINDOW_HEIGHT),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    # load saved stats (dataset log)
    try:
        with open(STATS_FILE) as dslog_file: 
            weathers = json.load(dslog_file)
    except Exception:
        weathers = []

    with SynchronyModel() as sync_mode:
        try:
            step = 0
            weather_idx = 1
            for weather, weather_dict in sync_mode.ds_counter.items():
                print("[{0:06d}] NEW WEATHER: {1}\n".format(sync_mode.captured_frame_no, weather))

                # resume where we left off or update the dataset stats adding a new entry
                if len(weathers) > weather_idx:
                    print("skipping this weather as it was already done")
                    continue
                elif len(weathers) < weather_idx:
                    weathers.append({
                        "params": str(weather), 
                        "start_idx" : int(sync_mode.captured_frame_no), 
                        "end_idx" : -1, 
                        "locations" : []
                        })

                sync_mode.set_weather(weather)

                for loc_idx, location in enumerate(weather_dict.keys()):

                    # Resume where we left off or update the dataset stats adding a new entry
                    if len(weathers[-1]["locations"]) > loc_idx+1 \
                        or (len(weathers[-1]["locations"]) == loc_idx+1 and weathers[-1]["locations"][-1]["end_idx"]) > 0:
                        print("skipping this location as it was already done")
                        continue
                    elif len(weathers[-1]["locations"]) < loc_idx+1:
                        print("Adding new entry for location", loc_idx+1)
                        weathers[-1]["locations"].append({
                            "params" : str(location),
                            "start_idx" : int(sync_mode.captured_frame_no),
                            "end_idx" : -1,
                            "cur_idx" : int(sync_mode.captured_frame_no),
                            "ds_classes" : {
                                "Pedestrian" : 0,
                                "Car" : 0,
                                "Cyclist": 0
                            }
                        })

                    print("[{0:06d}] NEW LOCATION: {1}\n".format(sync_mode.captured_frame_no, location))
                    sync_mode.intrinsic, sync_mode.my_camera = sync_mode._span_spectator(location.location, location.rotation)
                    print("map name:", sync_mode.world.get_map().name)

                    # Stop when we reach minimum criteria (2K samples for each class for each weather condition)
                    stop_collecting_threshold = 2000/len(sync_mode.locations)
                    while min(weathers[-1]["locations"][-1]["ds_classes"].values()) < stop_collecting_threshold:
    
                        if should_quit(sync_mode):
                            break

                        if step % 60 is 0:
                            print(sync_mode.player.get_transform())

                        clock.tick()
                        snapshot, sync_mode.main_image, sync_mode.depth_image, sync_mode.point_cloud = sync_mode.tick(timeout=2.0)

                        image = image_converter.to_rgb_array(sync_mode.main_image)
                        sync_mode.extrinsic = np.mat(sync_mode.my_camera.get_transform().get_matrix())

                        images = [image]
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            futures = [executor.submit(sync_mode.generate_datapoints, img) for img in images]
                        
                        try:
                            image, datapoints = futures[-1].result()
                        except Exception as e:
                            print(e)
                            continue

                        if datapoints and step % args.ds_interval is 0:
                            # Keep the dataset classes balanced: avoid collecting data if no minority is present 
                            frame_dp = {"Pedestrian": 0, "Cyclist": 0, "Car": 0}
                            ds_imp = 0
                            for dp in datapoints:
                                frame_dp[dp.type] +=1
                            for k, v in frame_dp.items():
                                if weathers[-1]["locations"][-1]["ds_classes"][k] < stop_collecting_threshold and v > 0:
                                    ds_imp +=1 
                            if ds_imp == 0:
                                print("SKIPPING FRAME as it contains 0 useful classes as of now")
                                continue

                            data = np.copy(np.frombuffer(sync_mode.point_cloud.raw_data, dtype=np.dtype('f4')))
                            data = np.reshape(data, (int(data.shape[0] / 4), 4))
                            # Isolate the 3D data
                            points = data[:, :-1]
                            # transform to car space
                            # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
                            # points = np.dot(sync_mode.player.get_transform().get_matrix(), points.T).T
                            # points = points[:, :-1]
                            # points[:, 2] -= LIDAR_HEIGHT_POS

                            # save training files asynchronously
                            threading.Thread(target=sync_mode._save_training_files, args=(datapoints, points,)).start()
                            sync_mode.captured_frame_no += 1

                            # Increment dataset counters
                            for dp in datapoints:
                                weathers[-1]["locations"][-1]["ds_classes"][dp.type] += 1
                            weathers[-1]["locations"][-1]["cur_idx"] = sync_mode.captured_frame_no

                            print("\n", weathers, "\n")

                        step = step+1
                        fps = round(1.0 / snapshot.timestamp.delta_seconds)
                        draw_image(display, image)
                        display.blit(
                            font.render('% 5d FPS (real)' % clock.get_fps(), True, (255, 255, 255)),
                            (8, 10))
                        display.blit(
                            font.render('% 5d FPS (simulated)' % fps, True, (255, 255, 255)),
                            (8, 28))
                        pygame.display.flip()

                        with open(STATS_FILE, "w") as dslog_file: 
                            json.dump(weathers, dslog_file)

                    weathers[-1]["locations"][-1]["end_idx"] = sync_mode.captured_frame_no

                weathers[-1]["end_idx"] = sync_mode.captured_frame_no
                weather_idx += 1

        finally:
            print('destroying actors.')
            for actor in sync_mode.actor_list:
                actor.destroy()
            pygame.quit()
            print('done.')


if __name__ == '__main__':
    main()
