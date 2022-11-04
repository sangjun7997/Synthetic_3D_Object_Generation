import numpy as np
import math

def rotation_matrix(lidar, rotation):
    rotation_x = rotation[0]
    rotation_y = rotation[1]
    rotation_z = rotation[2]

    lidar_temp = lidar.copy()
    lidar_temp_x = lidar_temp[:, 0]
    lidar_temp_y = lidar_temp[:, 1]
    lidar_temp_z = lidar_temp[:, 2]

    lidar_temptemp_x = lidar_temp_x.copy()
    lidar_temptemp_y = lidar_temp_y.copy()
    lidar_temptemp_z = lidar_temp_z.copy()

    lidar_temp_x = lidar_temptemp_x * (math.cos(rotation_y) * math.cos(rotation_z)) + lidar_temptemp_y * (-math.cos(rotation_y) * math.sin(rotation_z)) + lidar_temptemp_z * math.sin(rotation_y)
    lidar_temp_y = lidar_temptemp_x * (math.cos(rotation_x) * math.sin(rotation_z) + math.sin(rotation_x) * math.sin(rotation_y) * math.cos(rotation_z)) + lidar_temptemp_y * (math.cos(rotation_x) * math.cos(rotation_z) - math.sin(rotation_x) * math.sin(rotation_y) * math.sin(rotation_z))  + lidar_temptemp_z * (- math.sin(rotation_x) * math.cos(rotation_y))
    lidar_temp_z = lidar_temptemp_x * (math.sin(rotation_x) * math.sin(rotation_z) - math.cos(rotation_x) * math.sin(rotation_y) * math.cos(rotation_z)) + lidar_temptemp_y * (math.sin(rotation_x) * math.cos(rotation_z) + math.cos(rotation_x) * math.sin(rotation_y) * math.sin(rotation_z)) + lidar_temptemp_z * (math.cos(rotation_x) * math.cos(rotation_y))

    lidar[:,0] = lidar_temp_x
    lidar[:,1] = lidar_temp_y
    lidar[:,2] = lidar_temp_z

    return lidar