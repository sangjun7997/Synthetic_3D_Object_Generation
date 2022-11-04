import pcl
import numpy as np
import os
import sys
import open3d as o3d
import random
import json
import time
from tqdm import tqdm
from module.ground_removal import Processor
import copy

def from_xyz_to_xyzi(points):
    row, column = points.shape
    intensity = np.zeros((row, 1), dtype = np.float32)
    points = np.concatenate((points, intensity), axis = 1)
    return points

def load_pc_from_pcd(input_path, file_name):
    p = pcl.load_XYZI(input_path + '/' + file_name + '.pcd')
    return np.array(list(p), dtype = np.float32)

def load_pc_from_bin(input_path, file_name):
    obj = np.fromfile(input_path + '/' + file_name + '.bin', dtype = np.float32).reshape(-1, 4)
    return obj

def save_pcd_from_pc(points, pcd_path):
    pc = pcl.PointCloud_PointXYZI(points)
    pcl.save(pc, pcd_path)

def save_bin_from_pc(points, bin_path):
    points.tofile(bin_path)

def load_SUSTech_label(input_path, file_num):
    with open(input_path + '/json/' + file_num + '.json', 'r') as f:
        label_json = json.load(f)
    return label_json

def load_KITTI_label(input_path, file_num):
    with open(input_path + '/txt/'+ file_num + '.txt') as f:
        label_txt = f.readlines()
    return label_txt

def load_label_data(input_path, file_num):
    label_json = load_SUSTech_label(input_path, file_num)
    label_txt = load_KITTI_label(input_path, file_num)

    return label_json, label_txt

def load_lidar_data(input_path, lidar_name):
    if lidar_name.split('.')[1] == 'bin':
        road_original = load_pc_from_bin(input_path, lidar_name.split('.')[0])
    elif lidar_name.split('.')[1] == 'pcd':
        road_original = load_pc_from_pcd(input_path, lidar_name.split('.')[0])

    return road_original

def load_object_data(path):
    object_list = []
    label_list = []
    for obj in os.listdir(path + '/lidar'):
        lidar_num = obj.split('.')[0]
        lidar = load_pc_from_pcd(path + '/lidar/', lidar_num)
        object_list.append(lidar)

        with open(path + '/label/' + lidar_num + '.json') as f:
            label = json.load(f)
        label_list.append(label)

    return object_list, label_list

def get_matrix(file, v2c, rect):
    with open(file) as f:
        lines = f.readlines()
        trans = [x for x in filter(lambda s: s.startswith(v2c), lines)][0]
        
        matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]
        matrix = matrix + [0,0,0,1]
        m = np.array(matrix)
        velo_to_cam  = m.reshape([4,4])

        trans = [x for x in filter(lambda s: s.startswith(rect), lines)][0]
        matrix = [m for m in map(lambda x: float(x), trans.strip().split(" ")[1:])]        
        m = np.array(matrix).reshape(3,3)
        
        m = np.concatenate((m, np.expand_dims(np.zeros(3), 1)), axis=1)
        
        rect = np.concatenate((m, np.expand_dims(np.array([0,0,0,1]), 0)), axis=0)        
        
        m = np.matmul(rect, velo_to_cam)
        
        return m

def make_filename_list(input_path, output_path):
    bin_name_list = []
    
    for input_bin in sorted(os.listdir(input_path)):
        bin_name = input_bin.split('.')[0]
        bin_name_list.append(bin_name)
    
    for output_pcd in os.listdir(output_path):
        pcd_num = output_pcd.split('.')[0]
        if pcd_num in bin_name_list:
            bin_name_list.remove(pcd_num)

    return bin_name_list