"""
This file contains all the methods responsible for saving the generated data in the correct output format.

"""

import numpy as np
import os
import logging
from utils import degrees_to_radians
import math
import pickle


def save_groundplanes(planes_fname, player, lidar_height):
    from math import cos, sin
    """ Saves the groundplane vector of the current frame.
        The format of the ground plane file is first three lines describing the file (number of parameters).
        The next line is the three parameters of the normal vector, and the last is the height of the normal vector,
        which is the same as the distance to the camera in meters.
    """
    rotation = player.get_transform().rotation
    pitch, roll = rotation.pitch, rotation.roll
    # Since measurements are in degrees, convert to radians
    pitch = degrees_to_radians(pitch)
    roll = degrees_to_radians(roll)
    # Rotate normal vector (y) wrt. pitch and yaw
    normal_vector = [cos(pitch)*sin(roll),
                     -cos(pitch)*cos(roll),
                     sin(pitch)
                     ]
    normal_vector = map(str, normal_vector)
    with open(planes_fname, 'w') as f:
        f.write("# Plane\n")
        f.write("Width 4\n")
        f.write("Height 1\n")
        f.write("{} {}\n".format(" ".join(normal_vector), lidar_height))
    logging.info("Wrote plane data to %s", planes_fname)


def save_ref_files(OUTPUT_FOLDER, id):
    """ Appends the id of the given record to the files """
    for name in ['train.txt', 'val.txt', 'trainval.txt']:
        path = os.path.join(OUTPUT_FOLDER, name)
        with open(path, 'a') as f:
            f.write("{0:06}".format(id) + '\n')
        logging.info("Wrote reference files to %s", path)


def save_image_data(filename, image):
    logging.info("Wrote image data to %s", filename)
    image.save_to_disk(filename)


def save_lidar_data(filename, point_cloud, format="bin"):
    """ Saves lidar data to given filename, according to the lidar data format.
        bin is used for KITTI-data format, while .ply is the regular point cloud format
        In Unreal, the coordinate system of the engine is defined as, which is the same as the lidar points
        z
        ^   ^ x
        |  /
        | /
        |/____> y
        This is a left-handed coordinate system, with x being forward, y to the right and z up
        See also https://github.com/carla-simulator/carla/issues/498
        However, the lidar coordinate system from KITTI is defined as
              z
              ^   ^ x
              |  /
              | /
        y<____|/
        Which is a right handed coordinate sylstem
        Therefore, we need to flip the y axis of the lidar in order to get the correct lidar format for kitti.
        This corresponds to the following changes from Carla to Kitti
            Carla: X   Y   Z
            KITTI: X  -Y   Z
        NOTE: We do not flip the coordinate system when saving to .ply.
    """
    logging.info("Wrote lidar data to %s", filename)

    if format == "bin":
        lidar_array = [[point[0], -point[1], point[2], 1.0]
                       for point in point_cloud]
        lidar_array = np.array(lidar_array).astype(np.float32)
        logging.debug("Lidar min/max of x: {} {}".format(
                      lidar_array[:, 0].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of y: {} {}".format(
                      lidar_array[:, 1].min(), lidar_array[:, 0].max()))
        logging.debug("Lidar min/max of z: {} {}".format(
                      lidar_array[:, 2].min(), lidar_array[:, 0].max()))
        lidar_array.tofile(filename)

    # point_cloud.save_to_disk(filename)




def save_kitti_data(filename, datapoints):
    with open(filename, 'w') as f:
        out_str = "\n".join([str(point) for point in datapoints if point])
        f.write(out_str)
    logging.info("Wrote kitti data to %s", filename)

def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]])

    return R

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    yaw=math.atan2(R[1,0], R[0,0])
    pitch=math.atan2(-R[2,0], math.sqrt(R[2,1]**2+R[2,2]**2))
    roll=math.atan2(R[2,1], R[2,2])
    return pitch, yaw, roll

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :

    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])

    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])

    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def save_calibration_matrices(filename, intrinsic_mat, extrinsic_mat, save_ie=True):
    """ Saves the calibration matrices to a file.
        AVOD (and KITTI) refers to P as P=K*[R;t], so we will just store P.
        The resulting file will contain:
        3x4    p0-p3      Camera P matrix. Contains extrinsic
                          and intrinsic parameters. (P=K*[R;t])
        3x3    r0_rect    Rectification matrix, required to transform points
                          from velodyne to camera coordinate frame.
        3x4    tr_velodyne_to_cam    Used to transform from velodyne to cam
                                     coordinate frame according to:
                                     Point_Camera = P_cam * R0_rect *
                                                    Tr_velo_to_cam *
                                                    Point_Velodyne.
        3x4    tr_imu_to_velo        Used to transform from imu to velodyne coordinate frame. This is not needed since we do not export
                                     imu data.
    """
    # KITTI format demands that we flatten in row-major order
    ravel_mode = 'C'
    P0 = intrinsic_mat
    print("INTR:", intrinsic_mat)
    print("EXTR:", extrinsic_mat)
    P0 = np.column_stack((P0, np.array([0, 0, 0])))

    R = extrinsic_mat[:3, :3]
    T = extrinsic_mat[:3, 3]
    T1 = T
    # T1 = np.array([T[0,0], -T[2,0], T[1,0]])
    # T1 = np.array([0, -T[2,0], 0])
    # T1 = np.array([0, 0, 0])

    pitch, yaw, roll = rotationMatrixToEulerAngles(R)
    # R1 = np.array(eulerAnglesToRotationMatrix((-roll, 0, 0)))
    # R1 = np.array(eulerAnglesToRotationMatrix((-pitch, 0, 0)))  # roll, pitch, yaw

    R1 = np.array(eulerAnglesToRotationMatrix((0, 0, 0)))  # roll, pitch, yaw
    
    RT1 = np.column_stack((R1, T1))
    RT1 = np.row_stack((RT1, np.array([0, 0, 0, 1])))
    print("RT1:", RT1)
    P0 = np.matmul(P0, RT1)

    # P0 = np.matmul(P0, extrinsic_mat)

    print("P0:", P0)
    P0 = np.ravel(P0, order=ravel_mode)
    print("P0:", P0)

    R0 = np.identity(3)
    TR_velodyne = np.array([[0, -1, 0],
                            [0, 0, -1],
                            [1, 0, 0]])
                            
    # Add translation vector from velo to camera. This is 0 because the position of camera and lidar is equal in our configuration.
    TR_velodyne = np.column_stack((TR_velodyne, np.array([0, 0, 0])))

    # TR_velodyne = np.linalg.inv(RT1)[:3, :4] # Debug

    TR_imu_to_velo = np.identity(3)
    TR_imu_to_velo = np.column_stack((TR_imu_to_velo, np.array([0, 0, 0])))

    def write_flat(f, name, arr):
        f.write("{}: {}\n".format(name, ' '.join(
            map(str, arr.flatten(ravel_mode).squeeze()))))

    if save_ie:
        with open(filename[:-3] + "i", 'wb') as f:
            np.save(f, intrinsic_mat)
        with open(filename[:-3] + "e", 'wb') as f:
            np.save(f, extrinsic_mat)

    # All matrices are written on a line with spacing
    with open(filename, 'w') as f:
        for i in range(4):  # Avod expects all 4 P-matrices even though we only use the first
            write_flat(f, "P" + str(i), P0)
        write_flat(f, "R0_rect", R0)
        write_flat(f, "Tr_velo_to_cam", TR_velodyne)
        write_flat(f, "TR_imu_to_velo", TR_imu_to_velo)
    print("Wrote all calibration matrices to %s", filename)