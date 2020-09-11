import random

import numpy as np
from scipy.spatial.transform import Rotation


def rotate_impact(robot_pose, sensor_pose):
    """
    Calculate impact pose in world coordinates.
    """

    # base to force sensor transformation
    trafo_sensor_in_robot = np.array([[1, 0, 0,     0],
                                      [0, 1, 0,     0],
                                      [0, 0, 1, 0.522],
                                      [0, 0, 0,     1]])

    # world to base transformation
    trafo_robot_in_world = np.array([[1, 0, 0, robot_pose[0]],
                                     [0, 1, 0, robot_pose[1]],
                                     [0, 0, 1, robot_pose[2]],
                                     [0, 0, 0,             1]])

    # rotation
    trafo_robot_in_world[0:3, 0:3] = Rotation.from_quat([robot_pose[3],
                                                         robot_pose[4],
                                                         robot_pose[5],
                                                         robot_pose[6]]).as_dcm()

    # impact pose in shell
    impact_in_sensor = np.array([[1, 0, 0, sensor_pose[0]],
                                 [0, 1, 0, sensor_pose[1]],
                                 [0, 0, 1, sensor_pose[2]],
                                 [0, 0, 0,              1]])

    # rotation
    impact_in_sensor[0:3, 0:3] = Rotation.from_quat([sensor_pose[3],
                                                     sensor_pose[4],
                                                     sensor_pose[5],
                                                     sensor_pose[6]]).as_dcm()

    # impact pose in world coordinates
    impact_in_world = np.dot(np.dot(trafo_robot_in_world, trafo_sensor_in_robot),
                             impact_in_sensor)

    # return rotated 2D pose as [x,y,theta]
    impact_in_world_x = impact_in_world[0, 3]
    impact_in_world_y = impact_in_world[1, 3]
    impact_in_world_theta = Rotation.from_dcm(impact_in_world[0:3, 0:3]).as_euler('xyz')[2]

    return [impact_in_world_x, impact_in_world_y, impact_in_world_theta]


def parse_pose_file(pose_file_path, map_path, map_handler):
    """
    Parse recorded collision pose file to generate pose list with collision labels.
    """

    pose_list = []

    pose_file = open(pose_file_path, "r")
    lines = pose_file.readlines()

    # parse lines, starting in line 9 after the file header
    for line in lines[9::]:

        data_line = [float(entry) for entry in line.strip().split(" ")]

        # extract the data
        robot_pose = data_line[0:7]
        impact_pose = data_line[7:14]
        collision = bool(data_line[15])

        target_poses = []

        # for a collision example, we save the pose of the collision in the
        # map frame
        if collision:

            target_pose = rotate_impact(robot_pose, impact_pose)
            # flip collision angle, because the impact points towards the robot
            target_pose[2] += np.pi
            target_poses.append(target_pose)

        # for a non-collision example, we save the front of the robot and a
        # random position inside the robot footprint, in the map frame
        else:

            inside_coords = []
            robot_radius = 0.275

            # add robot front as negative example
            inside_coords.append([robot_radius, 0.0])

            # sample random position inside robot footprint as negative example
            angle = random.random() * 2 * np.pi
            dist = np.sqrt(random.random()) * robot_radius
            inside_coords.append([np.cos(angle) * dist, np.sin(angle) * dist])
            inside_coords = np.array(inside_coords)

            for x, y in inside_coords:

                inside_pose = np.array([x, y, 0, 0, 0, 0, 1])
                target_pose = rotate_impact(robot_pose, inside_pose)
                target_poses.append(target_pose)

        for target_pose in target_poses:

            # check if target pose is inside the map
            cell_x, cell_y = map_handler.world_to_cell(target_pose[0], target_pose[1])
            if map_handler.inside_map(cell_x, cell_y):

                pose = {'pose': target_pose,
                        'collision': collision,
                        'map_file': map_path}

                pose_list.append(pose)

    return pose_list
