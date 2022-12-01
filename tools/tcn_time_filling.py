'''
Creating the file contains the per-frame action labels for training the tcn.
The original annotated file <tcn_time_point.npy> records the transition time points.
'''

import os

import numpy as np

import glob as glob

from params import *

tcn_time_overall = np.load("../"+tcn_time_dir, allow_pickle=True)

for i in range(len(tcn_time_overall)):
    time_points = tcn_time_overall[i]
    images_path = os.path.join(frames_dir, str("{:03d}").format(i+1), "*.jpg")
    num_images = len(glob.glob(images_path))
    time_points_filling = np.zeros([1, num_images])

    # 0-beginning 1-jumping 2-dropping 3-entering water 4-ending
    if len(time_points) == 4:
        time_points_filling[0, 0:time_points[0]] = 0
        time_points_filling[0, time_points[0]:time_points[1]] = 1
        time_points_filling[0, time_points[1]:time_points[2]] = 2
        time_points_filling[0, time_points[2]:time_points[3]] = 3
        time_points_filling[0, time_points[3]:] = 4

    # 1-jumping 2-dropping 3-entering water 4-ending, some videos can only be divided into four stages
    elif len(time_points) == 3:
        time_points_filling[0, 0:time_points[0]] = 1
        time_points_filling[0, time_points[0]:time_points[1]] = 2
        time_points_filling[0, time_points[1]:time_points[2]] = 3
        time_points_filling[0, time_points[2]:] = 4

    time_points_filling = time_points_filling.astype(int).tolist()

    save_path = os.path.join("../"+tcn_time_points_filling_save_dir, str("{:03d}").format(i+1)+'.npy')

    np.save(save_path, time_points_filling)

print("tcn label filling done")






