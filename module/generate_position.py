import numpy as np
import math
import os
import sys
import open3d as o3d
import random
import copy
from module.ground_removal import Processor
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataloader.loader import *
from tools.convert_coordinate import *

try:
    patchwork_module_path = os.path.join("/home/server10/Synthetic_3D_Object_Generation", "build/python_wrapper")
    sys.path.insert(0, patchwork_module_path)
    import pypatchworkpp
except ImportError:
    print("Cannot find pypatchworkpp!")
    exit(1)

def pose_estimation(position, label):
    rotation = np.zeros(position.shape, dtype = np.float32)

    # default position
    for pos_idx in range(position.shape[0]):
        # area 1, 5
        if (position[pos_idx, 1] < -30) or (30 < position[pos_idx, 1]):
            rotation[pos_idx, 2] = np.float32(random.uniform(0, 2 * math.pi))
        # area 2
        elif 10 < position[pos_idx, 1]:
            rot_1 = 0
            rot_2 = random.uniform(0, 2 * math.pi)
            rotation[pos_idx, 2] = np.float32(random.choices([rot_1, rot_2], [70, 30]))
        # area 4
        elif position[pos_idx, 1] < -10:
            rot_1 = math.pi
            rot_2 = random.uniform(0, 2 * math.pi)
            rotation[pos_idx, 2] = np.float32(random.choices([rot_1, rot_2], [70, 30]))
        else:
            rot_1 = 0
            rot_2 = math.pi
            rotation[pos_idx, 2] = np.float32(random.choice([rot_1, rot_2]))

    # near label
    for pos_idx in range(position.shape[0]):
        dist_min = 10

        object_position_x = position[pos_idx, 0]
        object_position_y = position[pos_idx, 1]
        object_position_z = position[pos_idx, 2]

        for lab in label:
            label_position_x = lab['psr']['position']['x']
            label_position_y = lab['psr']['position']['y']
            label_position_z = lab['psr']['position']['z']

            label_rotation_x = lab['psr']['rotation']['x']
            label_rotation_y = lab['psr']['rotation']['y']
            label_rotation_z = lab['psr']['rotation']['z']

            distance = math.sqrt((label_position_x - object_position_x) ** 2 + (label_position_y - object_position_y) ** 2 + (label_position_z - object_position_z) ** 2)

            if distance < dist_min:
                dist_min = distance
                rotation[pos_idx, 0] = label_rotation_x
                rotation[pos_idx, 1] = label_rotation_y
                rotation[pos_idx, 2] = label_rotation_z

    return rotation

def ground_filtering(frame):
    # Patchwork++ initialization
    params = pypatchworkpp.Parameters()
    params.verbose = False
    PatchworkPLUSPLUS = pypatchworkpp.patchworkpp(params)

    ptcloud_xyz = frame[:, :-1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptcloud_xyz)

    pointcloud = np.asarray(pcd.points)

    PatchworkPLUSPLUS.estimateGround(pointcloud)

    ground = PatchworkPLUSPLUS.getGround()
    nonground = PatchworkPLUSPLUS.getNonground()

    return ground, nonground

def object_collision_filtering(ground, collision_threshold):
    np.random.shuffle(ground)

    searching = True
    while searching:
        collision = False
        delete_list = []
        for idx1 in range(ground.shape[0]):
            for idx2 in range(ground.shape[0]):
                if idx1 == idx2:
                    continue
                distance = math.sqrt((ground[idx1, 0] - ground[idx2, 0])**2 + (ground[idx1, 1] - ground[idx2, 1])**2 + (ground[idx1, 2] - ground[idx2, 2])**2)
                if distance < collision_threshold:
                    ground = np.delete(ground, idx2, axis = 0)
                    collision = True
                    break
            if collision == True:
                break
        if collision == False:
            break

    return ground


def collision_filtering(ground, obj_point, collision_threshold):
    ground_x = ground[:, 0]
    ground_y = ground[:, 1]

    obj_x = obj_point[:, 0]
    obj_y = obj_point[:, 1]

    delete_list = []

    for pos in range(ground.shape[0]):
        for obj in range(obj_point.shape[0]):
            distance = math.sqrt(math.pow(ground_x[pos] - obj_x[obj], 2) + math.pow(ground_y[pos] - obj_y[obj], 2))
            if distance < collision_threshold:
                delete_list.append(pos)

    ground = np.delete(ground, delete_list, axis = 0)

    return ground


def select_position(ground, nonground, label_json, min_distance, max_distance, max_object_num, sampling_num, select_epoch, collision_threshold):
    print("-- Generating Object position...")

    ground = from_xyz_to_xyzi(ground)
    ground = np.delete(ground, np.where(ground[:, 0] < 0), axis = 0)
    ground_spherical = convert_orthogonal_to_spherical(ground)
    ground = np.delete(ground, np.where(ground_spherical[:, 0] < 10), axis=0)

    object_num = 1
    position = np.zeros((1, 3), dtype = np.float32)
    position_zero = np.zeros((1, 3), dtype = np.float32)

    if sampling_num < ground.shape[0]:
        for epoch in range(select_epoch):
            random_index = random.sample(range(ground.shape[0]), ground.shape[0] - sampling_num)
            ground_sample = np.delete(ground, random_index, axis = 0)

            ground_sample = np.delete(ground_sample, np.where(np.power(ground_sample[:, 0], 2) + np.power(ground_sample[:, 1], 2) < collision_threshold * collision_threshold), axis = 0)

            if ground_sample.shape[0] == 0:
                ground_sample = None
            else:
                ground_sample = object_collision_filtering(ground_sample, collision_threshold)

            if ground_sample.shape[0] == 0:
                ground_sample = None
            else:
                ground_sample = collision_filtering(ground_sample, nonground, collision_threshold / 2)

            if np.array_equal(position, position_zero) and object_num <= ground_sample.shape[0]:
                object_num = ground_sample.shape[0]
                position = ground_sample.copy()
            elif object_num < ground_sample.shape[0]:
                object_num = ground_sample.shape[0]
                position = ground_sample.copy()

        if np.array_equal(position, position_zero):
            position = None
        else:
            label_num = len(label_json)
            
            if label_num > 0:
                prev_label_list = np.array([[label_json[0]['psr']['position']['x'], label_json[0]['psr']['position']['y']]])
            else:
                prev_label_list = []
            
            for label_idx in range(label_num - 1):
                prev_label_list = np.concatenate((prev_label_list, np.array([[label_json[label_idx + 1]['psr']['position']['x'], label_json[label_idx + 1]['psr']['position']['y']]])), axis = 0)
            
            if label_num > 0:
                position = collision_filtering(ground_sample, prev_label_list, collision_threshold)
            else:
                position = ground_sample

        if ground_sample.shape[0] == 0:
            ground_sample = None
        else:
            if max_object_num < ground_sample.shape[0]:
                ground_sample = ground_sample[:max_object_num]

    else:
        position = None

    if position is not None:
        print("   Position : \n{}".format(position))
    return position