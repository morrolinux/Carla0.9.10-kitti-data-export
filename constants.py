""" DATA GENERATION SETTINGS"""
GEN_DATA = True  # Whether or not to save training data
VISUALIZE_LIDAR = True
# How many frames to wait between each capture of screen, bounding boxes and lidar
STEPS_BETWEEN_RECORDINGS = 10
CLASSES_TO_LABEL = ["Vehicle", "Pedestrian"]
# Lidar can be saved in bin to comply to kitti, or the standard .ply format
LIDAR_DATA_FORMAT = "bin"
assert LIDAR_DATA_FORMAT in [
    "bin", "ply"], "Lidar data format must be either bin or ply"
OCCLUDED_VERTEX_COLOR = (255, 0, 0)
VISIBLE_VERTEX_COLOR = (0, 255, 0)
# How many meters the car must drive before a new capture is triggered.
DISTANCE_SINCE_LAST_RECORDING = 10
# How many datapoints to record before resetting the scene.
NUM_RECORDINGS_BEFORE_RESET = 20
# How many frames to render before resetting the environment
# For example, the agent may be stuck
NUM_EMPTY_FRAMES_BEFORE_RESET = 100

""" CARLA SETTINGS """
CAMERA_HEIGHT_POS = 0
LIDAR_HEIGHT_POS = CAMERA_HEIGHT_POS
MIN_BBOX_AREA_IN_PX = 80


""" AGENT SETTINGS """
NUM_VEHICLES = 80
NUM_PEDESTRIANS = 40

""" RENDERING SETTINGS """
# WINDOW_WIDTH = 1248
# WINDOW_HEIGHT = 384
WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080

MINI_WINDOW_WIDTH = 320
MINI_WINDOW_HEIGHT = 180

WINDOW_WIDTH_HALF = WINDOW_WIDTH / 2
WINDOW_HEIGHT_HALF = WINDOW_HEIGHT / 2

MAX_RENDER_DEPTH_IN_METERS = 90  # Meters
MIN_VISIBLE_VERTICES_FOR_RENDER = 5

NUM_OF_WALKERS = 256
FILTERW = 'walker.pedestrian.*'
FILTERV = 'vehicle.*'
NUM_OF_VEHICLES = 256