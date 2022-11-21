import numpy as np
import os
import sys
import time
from tqdm import tqdm

from tools.convert_coordinate import *
from tools.rotate_points import *
from dataloader.loader import *

from module.generate_object import *
from module.generate_label import *
from module.generate_position import *

#np.set_printoptions(threshold=np.inf, linewidth=np.inf)

h_resolution = 0.0060       # horizontal resolution (assuming 20Hz setting)
#v_resolution = 0.0055       # vertical res 64ch
v_resolution = 0.0030       # vertical res 128ch
max_object_num = 10
object_sampling_num = 100
min_object_distance = 10
max_object_distance = 100
collision_threshold = 6
select_epoch = 10

input_lidar_path = '/media/server10/T7/LiDAR_bag/Synth/lidar'
input_label_path = '/media/server10/T7/LiDAR_bag/Synth/label'
input_calib_path = '/media/server10/T7/LiDAR_bag/Synth/calib'

output_lidar_path = '/media/server10/T7/SynthPoints/20221121_output/lidar'
output_label_path = '/media/server10/T7/SynthPoints/20221121_output/label'

object_path = 'data/object/car'

object_lidar_list, object_label_list = load_object_data(object_path)

bin_name_list = make_filename_list(input_lidar_path, output_lidar_path)

total_new_object = 0
total_delete_object = 0
 
start = time.time()

#for lidar in tqdm(sorted(os.listdir(lidar_path))):
#for lidar in tqdm(bin_name_list):
for lidar in bin_name_list:
    print("-----------------------------------------------------------------------")
    print("File Num : {}".format(lidar))
    lidar = lidar + '.pcd'
    lidar_num = lidar.split('.')[0]

    road_original = load_lidar_data(input_lidar_path, lidar)
    road_original = np.delete(road_original, np.where((road_original[:, 0] == 0) & (road_original[:, 1] == 0) & (road_original[:, 2] == 0)), axis = 0)

    label_json, label_txt = load_label_data(input_label_path, lidar_num)

    calib_matrix = get_matrix(os.path.join(input_calib_path, lidar_num + '.txt'), "Tr_velo_to_cam", "R0_rect")

    prev_object_num = len(label_json)

    ground, nonground = ground_filtering(road_original)
    position = select_position(ground, nonground, label_json, min_object_distance, max_object_distance, max_object_num, object_sampling_num, select_epoch, collision_threshold)

    if position is None:
        save_bin_from_pc(road_original, output_lidar_path + '/' + lidar_num + '.bin')
        with open(output_label_path + '/' + lidar_num + '.txt', 'w') as outfile:
            outfile.writelines(label_txt)

    else:
        road_temp = road_original.copy()
        json_label_temp = label_json.copy()
        txt_label_temp = label_txt.copy()

        rotation = pose_estimation(position, json_label_temp)
        
        for obj_num in range(position.shape[0]):
            rand_idx = random.randrange(len(object_lidar_list))
            obj_lidar = object_lidar_list[rand_idx]
            obj_label = object_label_list[rand_idx]
            road_temp, object_point, occluded_object_point = generate_object(road_temp, obj_lidar, obj_label, position[obj_num, :], rotation[obj_num, :], h_resolution, v_resolution)
            txt_label_temp = add_KITTI_label_data(object_point, occluded_object_point, txt_label_temp, obj_label, position[obj_num, :], rotation[obj_num, :], calib_matrix)
            total_new_object = total_new_object + 1    

        txt_label_copy = copy.deepcopy(txt_label_temp)
        notDontCares = [x for x in txt_label_temp if x.strip().split(" ")[0] != "DontCare"]
        DontCares = [x for x in txt_label_copy if x.strip().split(" ")[0] == "DontCare"] 

        notDontCares, delete_num = check_object_points(road_temp, notDontCares, calib_matrix)
        total_delete_object = total_delete_object + delete_num

        filtered_objs = notDontCares + DontCares

        save_bin_from_pc(road_temp, output_lidar_path + '/' + lidar_num + '.bin')

        with open(output_label_path + '/' + lidar_num + '.txt', 'w') as outfile:
            outfile.writelines(filtered_objs)

second = time.time() - start

print("Total Processing time : {}".format(second))
print("Generated Object Num : {}".format(total_new_object))
print("Deleted Object Num : {}".format(total_delete_object))