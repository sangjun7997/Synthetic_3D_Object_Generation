import os
import sys
import json
import time
import numpy as np
import math

input_label_path = '/home/sangjun/Desktop/SynthPoints/output/label'
input_label_path = '/media/sangjun/T7/Deep_Learning_Dataset/KITTI_3D_OD/data_object_label_2/training/label_2'
output_label_path = '/home/sangjun/Desktop/SynthPoints/output/new_label'
src_calib_file = '/home/sangjun/Desktop/SynthPoints/output/calib/000000.txt'

class_list = []
class_num = []

dist_list = ['~10', '~20', '~30', '~40', '~50', '~60', '~70', '~80', '~90', '~100']
dist_num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

def get_inv_matrix(file, v2c, rect):
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

        m = np.linalg.inv(m)
        
        return m

for file_name in os.listdir(input_label_path):
    with open(input_label_path + '/' + file_name, 'r' ) as label_file:
        lines = label_file.readlines()
    for line_num, line in enumerate(lines):
        line_split = line.split()
        there_is = False
        for list_num in range(len(class_list)):
            if class_list[list_num] == line_split[0]:
                there_is = True
                class_num[list_num] = class_num[list_num] + 1
        if there_is == False:
            class_list.append(line_split[0])
            class_num.append(1)

    inv_matrix = get_inv_matrix(src_calib_file, "Tr_velo_to_cam", "R0_rect")

    for line_num, line in enumerate(lines):
        words = line.strip().split(" ")
        if words[0] == 'Car':
            pos = np.array([float(words[11]), float(words[12]), float(words[13]), 1]).T
            trans_pos = np.matmul(inv_matrix, pos)
            x = trans_pos[0]
            y = trans_pos[1]
            dist = math.sqrt(x**2 + y**2)
            if dist < 10:
                dist_num[0] = dist_num[0] + 1
            elif dist < 20:
                dist_num[1] = dist_num[1] + 1
            elif dist < 30:
                dist_num[2] = dist_num[2] + 1
            elif dist < 40:
                dist_num[3] = dist_num[3] + 1
            elif dist < 50:
                dist_num[4] = dist_num[4] + 1
            elif dist < 60:
                dist_num[5] = dist_num[5] + 1
            elif dist < 70:
                dist_num[6] = dist_num[6] + 1
            elif dist < 80:
                dist_num[7] = dist_num[7] + 1
            elif dist < 90:
                dist_num[8] = dist_num[8] + 1
            else:
                dist_num[9] = dist_num[9] + 1

    label_file.close()

print(class_list)
print(class_num)
print(dist_list)
print(dist_num)
