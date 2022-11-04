import numpy as np
import math
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from tools.convert_coordinate import *

def add_SUSTech_label_data(json_label, car_label, position, rotation):
    new_data = {
                'obj_id' : '0',
                'obj_type' : car_label[0]['obj_type'],
                'psr' : {
                    'position' : {
                        'x' : car_label[0]['psr']['position']['x'] + position[0],
                        'y' : car_label[0]['psr']['position']['y'] + position[1],
                        'z' : car_label[0]['psr']['position']['z'] + position[2]
                    },
                    'rotation' : {
                        'x' : car_label[0]['psr']['rotation']['x'] + rotation[0],
                        'y' : car_label[0]['psr']['rotation']['y'] + rotation[1],
                        'z' : car_label[0]['psr']['rotation']['z'] + rotation[2]
                    },
                    'scale' : {
                        'x' : car_label[0]['psr']['scale']['x'],
                        'y' : car_label[0]['psr']['scale']['y'],
                        'z' : car_label[0]['psr']['scale']['z']
                    }
                }
            }
    json_label.append(new_data)
    return json_label

def add_KITTI_label_data(object_point1, object_point2, txt_label, car_label, position, rotation, calib_matrix):
    original_object_spherical = convert_orthogonal_to_spherical(object_point1)
    occluded_object_spherical = convert_orthogonal_to_spherical(object_point2)

    if object_point2.shape[0] < 5:
        return txt_label

    original_object_pi = original_object_spherical[:, 2]
    original_object_pi_max = np.max(original_object_pi)
    original_object_pi_min = np.min(original_object_pi)

    if original_object_pi_max - original_object_pi_min < math.pi:
        original_distance = original_object_pi_max - original_object_pi_min
    else:
        original_distance = original_object_pi_min + (2 * math.pi - original_object_pi_max)

    occluded_object_pi = occluded_object_spherical[:, 2]
    occluded_object_pi_max = np.max(occluded_object_pi)
    occluded_object_pi_min = np.min(occluded_object_pi)

    if occluded_object_pi_max - occluded_object_pi_min < math.pi:
        occluded_distance = occluded_object_pi_max - occluded_object_pi_min
    else:
        occluded_distance = occluded_object_pi_min + (2 * math.pi - occluded_object_pi_max)

    occluded = 1 - occluded_distance / original_distance

    if occluded < 0.3:
        occluded = 0
    elif occluded < 0.6:
        occluded = 1
    else:
        occluded = 2
    
    bbox = [804.79, 167.34, 995.43, 327.94]

    pos = np.array([car_label[0]["psr"]["position"]["x"] + position[0], car_label[0]["psr"]["position"]["y"] + position[1], car_label[0]["psr"]["position"]["z"] + position[2] - car_label[0]["psr"]["scale"]["z"] / 2, 1]).T
    trans_pos = np.matmul(calib_matrix, pos)
    location = [trans_pos[0], trans_pos[1], trans_pos[2]]

    dimensions = [car_label[0]["psr"]["scale"]["z"], car_label[0]["psr"]["scale"]["y"], car_label[0]["psr"]["scale"]["x"]]

    rotation_y = - (car_label[0]["psr"]["rotation"]["z"] + rotation[2]) - math.pi / 2

    alpha = rotation_y

    obj_type = car_label[0]['obj_type']

    obj = str(obj_type) + " " + "0.00" + " " + str(occluded) + " " + str(round(alpha, 2)) + " " + str(round(float(bbox[0]), 2)) + " " + str(round(float(bbox[1]), 2)) + " " + str(round(float(bbox[2]), 2)) + " " + str(round(float(bbox[3]), 2)) + " " + str(round(float(dimensions[0]),2)) + " " + str(round(float(dimensions[1]),2)) + " " + str(round(float(dimensions[2]),2)) + " " + str(round(float(location[0]),2)) + " " + str(round(float(location[1]),2)) + " " + str(round(float(location[2]),2)) + " " + str(round(rotation_y, 2)) + "\n"

    txt_label.append(obj)

    return txt_label