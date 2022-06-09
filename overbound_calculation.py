from itertools import count
import numpy as np
from sklearn import datasets
from src.io import npy_events_tools, dat_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import h5py
import pickle
import torch
import time
import math

raw_dir = "/data2/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"

under_zero_x = []
under_zero_y = []
over_bound_x = []
over_bound_y = []

for mode in ["train","test","val"]:
    
    file_dir = os.path.join(raw_dir, mode)
    root = file_dir
    #h5 = h5py.File(raw_dir + '/ATIS_taf_'+mode+'.h5', 'w')
    try:
        files = os.listdir(file_dir)
    except Exception:
        continue
    # Remove duplicates (.npy and .dat)
    files = [time_seq_name[:-7] for time_seq_name in files
                    if time_seq_name[-3:] == 'dat']

    pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

    for i_file, file_name in enumerate(files):
        bbox_file = os.path.join(root, file_name + '_bbox.npy')
        # if os.path.exists(volume_save_path):
        #     continue
        #h5 = h5py.File(volume_save_path, "w")
        #f_bbox = open(new_bbox_file, "rb")
        f_bbox = open(bbox_file, "rb")
        start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        xs = dat_bbox['x']
        ys = dat_bbox['y']
        hs = dat_bbox['h']
        ws = dat_bbox['w']

        for x in xs:
            if ((x<0) and (x not in under_zero_x)):
                under_zero_x.append(x)
        for y in ys:
            if ((y<0) and (y not in under_zero_y)):
                under_zero_y.append(y)
        for h in hs:
            if ((y+h >= 240) and (h not in over_bound_y)):
                over_bound_y.append(y+h)
        for w in ws:
            if ((w+x >= 304) and (w not in over_bound_x)):
                over_bound_x.append(x+w)
        pbar.update()
    pbar.close()
print(under_zero_x,under_zero_y,over_bound_x,over_bound_y)
#print(under_zero_x.min(),under_zero_y.min(),over_bound_x.max(),over_bound_y.max())