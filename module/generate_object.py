import numpy as np
import math
import os
import sys
import random
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tools.rotate_points import *
from tools.convert_coordinate import *

def make_point_wall(theta_min, theta_max, pi_min, pi_max, h_res, v_res):
    wall_theta_t = np.arange(theta_min, theta_max, v_res, dtype = np.float32)
    wall_theta_t = np.array([wall_theta_t])
    wall_pi_t = np.arange(pi_min, pi_max, h_res / 2, dtype = np.float32)
    wall_pi_t = np.array([wall_pi_t])

    wall_theta_temp = wall_theta_t.T
    wall_pi_temp = wall_pi_t.T

    theta_row = wall_theta_temp.shape[0]
    pi_row = wall_pi_temp.shape[0]

    wall_theta_zero = np.full(wall_pi_temp.shape, wall_theta_temp[0,0], dtype = np.float32)
    wall_theta_pi_temp = np.concatenate((wall_theta_zero, wall_pi_temp), axis = 1)

    for theta in range(1, theta_row):
        wall_theta_temptemp = np.full(wall_pi_temp.shape, wall_theta_temp[theta, 0], dtype = np.float32)
        wall_theta_pi_temptemp = np.concatenate((wall_theta_temptemp, wall_pi_temp), axis = 1)
        wall_theta_pi_temp = np.concatenate((wall_theta_pi_temp, wall_theta_pi_temptemp), axis = 0)

    wall_r = np.full((wall_theta_pi_temp.shape[0], 1), 100, dtype = np.float32)
    wall_i = np.full((wall_theta_pi_temp.shape[0], 1), 0, dtype = np.float32)

    wall_spherical = np.concatenate((wall_r, wall_theta_pi_temp, wall_i), axis = 1)

    wall_row = wall_spherical.shape[0]
    wall_theta_noise = np.random.normal(0, 0.0001, (wall_row, 1))
    wall_pi_noise = np.random.normal(0, 0.0005, (wall_row, 1))
    wall_noise = np.concatenate((wall_theta_noise, wall_pi_noise), axis = 1)

    wall_spherical[:,1:3] = wall_spherical[:,1:3] + wall_noise

    wall_orthogonal = convert_spherical_to_orthogonal(wall_spherical)
    return wall_orthogonal

def sampling_object_points_by_distance(obj_point, position):
    distance = math.sqrt(position[0]**2 + position[1]**2 + position[2]**2)
    min_point_ratio = [95, 90, 80, 70, 60, 50]
    min_point_num = [80, 30, 30, 20, 10, 10]

    if distance < 30:
        if min_point_num[0] < obj_point.shape[0]:
            point_num = int((obj_point.shape[0] / 100) * min_point_ratio[0])
            obj_point = np.delete(obj_point, random.sample(range(0, obj_point.shape[0]), obj_point.shape[0] - point_num), axis = 0)
    elif distance < 50:
        if min_point_num[1] < obj_point.shape[0]:
            point_num = int((obj_point.shape[0] / 100) * min_point_ratio[1])
            obj_point = np.delete(obj_point, random.sample(range(0, obj_point.shape[0]), obj_point.shape[0] - point_num), axis = 0)
    elif distance < 60:
        if min_point_num[2] < obj_point.shape[0]:
            point_num = int((obj_point.shape[0] / 100) * min_point_ratio[2])
            obj_point = np.delete(obj_point, random.sample(range(0, obj_point.shape[0]), obj_point.shape[0] - point_num), axis = 0)
    elif distance < 70:
        if min_point_num[3] < obj_point.shape[0]:
            point_num = int((obj_point.shape[0] / 100) * min_point_ratio[3])
            obj_point = np.delete(obj_point, random.sample(range(0, obj_point.shape[0]), obj_point.shape[0] - point_num), axis = 0)
    elif distance < 80:
        if min_point_num[4] < obj_point.shape[0]:
            point_num = int((obj_point.shape[0] / 100) * min_point_ratio[4])
            obj_point = np.delete(obj_point, random.sample(range(0, obj_point.shape[0]), obj_point.shape[0] - point_num), axis = 0)
    elif distance < 100:
        if min_point_num[5] < obj_point.shape[0]:
            point_num = int((obj_point.shape[0] / 100) * min_point_ratio[5])
            obj_point = np.delete(obj_point, random.sample(range(0, obj_point.shape[0]), obj_point.shape[0] - point_num), axis = 0)

    return obj_point



def generate_object(road_orthogonal, object_orthogonal, object_label, position, rotation, h_res, v_res):
    #print("-- Generating Object on the Road in position [{:.3f}, {:.3f}, {:.3f}]...".format(position[0], position[1], position[2]))

    # Insert Object Position
    object_copy = object_orthogonal.copy()

    object_copy[:, 0] = object_copy[:, 0] - object_label[0]['psr']['position']['x']
    object_copy[:, 1] = object_copy[:, 1] - object_label[0]['psr']['position']['y']
    object_copy[:, 2] = object_copy[:, 2] - object_label[0]['psr']['position']['z']

    object_copy = rotation_matrix(object_copy, rotation)

    object_copy[:, 0] = object_copy[:, 0] + object_label[0]['psr']['position']['x']
    object_copy[:, 1] = object_copy[:, 1] + object_label[0]['psr']['position']['y']
    object_copy[:, 2] = object_copy[:, 2] + object_label[0]['psr']['position']['z']


    object_copy[:, 0] = object_copy[:, 0] + position[0]
    object_copy[:, 1] = object_copy[:, 1] + position[1]
    object_copy[:, 2] = object_copy[:, 2] + position[2]

    road_spherical = convert_orthogonal_to_spherical(road_orthogonal)
    object_spherical = convert_orthogonal_to_spherical(object_copy)
    
    road_theta = road_spherical[:, 1]
    road_pi = road_spherical[:, 2]

    object_num = object_spherical.shape[0]

    object_r = object_spherical[:, 0]
    object_theta = object_spherical[:, 1]
    object_pi = object_spherical[:, 2]

    object_theta_min = np.min(object_theta)
    object_theta_max = np.max(object_theta)
    object_pi_min = np.min(object_pi)
    object_pi_max = np.max(object_pi)

    if object_pi_max - object_pi_min < math.pi:
        object_left_pi = object_pi_max
        object_right_pi = object_pi_min
    else:
        object_spherical_left = np.delete(object_pi, np.where(math.pi < object_pi), 0)
        object_spherical_right = np.delete(object_pi, np.where(object_pi < math.pi), 0)
        object_left_pi = np.max(object_spherical_left)
        object_right_pi = np.min(object_spherical_right)

    # Seperate Road Points into RoI and Backgrouond
    if object_pi_max - object_pi_min < math.pi:
        roi_spherical = np.delete(road_spherical, np.where((road_theta < object_theta_min) | (object_theta_max < road_theta) | (road_pi < object_right_pi) | (object_left_pi < road_pi)), 0)
        bgd_spherical = np.delete(road_spherical, np.where((object_theta_min < road_theta) & (road_theta < object_theta_max) & (object_right_pi < road_pi) & (road_pi < object_left_pi)), 0)
    else:
        roi_spherical_left = np.delete(road_spherical, np.where((road_theta < object_theta_min) | (object_theta_max < road_theta) | (object_left_pi < road_pi)), 0)
        roi_spherical_right = np.delete(road_spherical, np.where((road_theta < object_theta_min) | (object_theta_max < road_theta) | (road_pi < object_right_pi)), 0)
        roi_spherical = np.concatenate((roi_spherical_left, roi_spherical_right), axis = 0)
        bgd_spherical = np.delete(road_spherical, np.where((object_theta_min < road_theta) & (road_theta < object_theta_max) & ((road_pi < object_left_pi) | (object_right_pi < road_pi))), 0)

    # Make Wall Points for RoI
    if object_pi_max - object_pi_min < math.pi:
        wall_orthogonal = make_point_wall(object_theta_min, object_theta_max, object_pi_min, object_pi_max, h_res, v_res)
    else:
        wall_orthogonal_left = make_point_wall(object_theta_min, object_theta_max, 0, object_left_pi, h_res, v_res)
        wall_orthogonal_right = make_point_wall(object_theta_min, object_theta_max, object_right_pi, 2 * math.pi, h_res, v_res)
        wall_orthogonal = np.concatenate((wall_orthogonal_left, wall_orthogonal_right), axis = 0)

    wall_spherical = convert_orthogonal_to_spherical(wall_orthogonal)
    
    wall_num = wall_spherical.shape[0]
    wall_r = wall_spherical[:, 0]
    wall_theta = wall_spherical[:, 1]
    wall_pi = wall_spherical[:, 2]

    #print("     Extracting Object Points from Wall Points...")

    for wall_point in wall_spherical:
        for object_point in object_spherical:
            h_distance = abs(wall_point[2]- object_point[2])
            v_distance = abs(wall_point[1]- object_point[1])
            if h_distance < h_res and v_distance < v_res and object_point[0] < wall_point[0]:
                wall_point[0] = object_point[0]
    projected_object = np.delete(wall_spherical, np.where(wall_r > 95), 0)
    projected_object_copy = projected_object.copy()
    projected_object_copy = convert_spherical_to_orthogonal(projected_object_copy)

    #print("   Done")

    projected_object_r = projected_object[:, 0]
    projected_object_theta = projected_object[:, 1]
    projected_object_pi = projected_object[:, 2]

    projected_object_num = projected_object.shape[0]

    roi_num = roi_spherical.shape[0]

    roi_spherical_r = roi_spherical[:, 0]
    roi_spherical_theta = roi_spherical[:, 1]
    roi_spherical_pi = roi_spherical[:, 2]

    roi_delete_list = []
    object_delete_list = []

    for roi_ind in range(roi_num):
        for object_ind in range(projected_object_num):
            h_distance = abs(roi_spherical_pi[roi_ind] - projected_object_pi[object_ind])
            v_distance = abs(roi_spherical_theta[roi_ind] - projected_object_theta[object_ind])

            if h_distance < h_res / 2 and v_distance < v_res / 2:
                if projected_object_r[object_ind] < roi_spherical_r[roi_ind]:
                    roi_delete_list.append(roi_ind)
                else:
                    object_delete_list.append(object_ind)
    
    roi_spherical = np.delete(roi_spherical, roi_delete_list, axis = 0)
    projected_object = np.delete(projected_object, object_delete_list, axis = 0)

    projected_object = sampling_object_points_by_distance(projected_object, position)

    road = np.concatenate((roi_spherical, bgd_spherical, projected_object), axis = 0)
    road = convert_spherical_to_orthogonal(road)

    projected_object = convert_spherical_to_orthogonal(projected_object)

    return road, projected_object_copy, projected_object

def check_object_points(lidar, label, calib_matrix):
    inv_calib_matrix = np.linalg.inv(calib_matrix)
    new_label = []
    delete_num = 0
    for obj in label:
        lidar_copy = lidar.copy()
        words = obj.strip().split(" ")
        pos = np.array([float(words[11]), float(words[12]), float(words[13]), 1]).T
        trans_pos = np.matmul(inv_calib_matrix, pos)

        position = [trans_pos[0], trans_pos[1], trans_pos[2] + float(words[8]) / 2]
        rotation = [0, 0, - math.pi / 2 - float(words[14])]
        scale = [float(words[10]), float(words[9]), float(words[8])]

        lidar_copy[:, 0] = lidar_copy[:, 0] - position[0]
        lidar_copy[:, 1] = lidar_copy[:, 1] - position[1]
        lidar_copy[:, 2] = lidar_copy[:, 2] - position[2]

        lidar_copy = rotation_matrix(lidar_copy, [0, 0, -rotation[2]])

        object_points = np.delete(lidar_copy, np.where((lidar_copy[:, 0] < -(scale[0] / 2)) | (scale[0] / 2 < lidar_copy[:, 0]) | (lidar_copy[:, 1] < -(scale[1] / 2)) | (scale[1] / 2 < lidar_copy[:, 1]) | (lidar_copy[:, 2] < -(scale[2] / 2)) | (scale[2] / 2 < lidar_copy[:, 2])), axis = 0)

        if 5 < object_points.shape[0]:
            new_label.append(obj)
        else:
            delete_num = delete_num + 1

    return new_label, delete_num

def get_points_in_bbox(lidar, label, calib_matrix):
    inv_calib_matrix = np.linalg.inv(calib_matrix)
    new_label = []
    delete_num = 0
    for obj in label:
        lidar_copy = lidar.copy()
        words = obj.strip().split(" ")
        pos = np.array([float(words[11]), float(words[12]), float(words[13]), 1]).T
        trans_pos = np.matmul(inv_calib_matrix, pos)

        position = [trans_pos[0], trans_pos[1], trans_pos[2] + float(words[8]) / 2]
        rotation = [0, 0, - math.pi / 2 - float(words[14])]
        scale = [float(words[10]), float(words[9]), float(words[8])]

        lidar_copy[:, 0] = lidar_copy[:, 0] - position[0]
        lidar_copy[:, 1] = lidar_copy[:, 1] - position[1]
        lidar_copy[:, 2] = lidar_copy[:, 2] - position[2]

        lidar_copy = rotation_matrix(lidar_copy, [0, 0, -rotation[2]])

        object_points = np.delete(lidar_copy, np.where((lidar_copy[:, 0] < -(scale[0] / 2)) | (scale[0] / 2 < lidar_copy[:, 0]) | (lidar_copy[:, 1] < -(scale[1] / 2)) | (scale[1] / 2 < lidar_copy[:, 1]) | (lidar_copy[:, 2] < -(scale[2] / 2)) | (scale[2] / 2 < lidar_copy[:, 2])), axis = 0)

    return object_points