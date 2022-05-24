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

raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset"
target_dir = "/data/lbd/Large_Automotive_Detection_Dataset"

for mode in ["train"]:
    
    file_dir = os.path.join(raw_dir, mode)
    root = file_dir
    target_root = os.path.join(target_dir, mode)
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
        new_bbox_file = os.path.join(target_root, file_name + '_bbox.npy')
        # if os.path.exists(volume_save_path):
        #     continue
        #h5 = h5py.File(volume_save_path, "w")
        #f_bbox = open(new_bbox_file, "rb")
        f_bbox = open(bbox_file, "rb")
        start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

        #f_event = psee_loader.PSEELoader(new_event_file)
        #print(unique_ts)
        #raise Exception("break")

        sampled_bboxes = []
        time_upperbound = -1e16
        for bbox_count,unique_time in enumerate(unique_ts):
            if unique_time <= 500000:
                continue
            if (unique_time - time_upperbound < 1000000):
                continue
            end_time = int(unique_time)
            
            sampled_bboxes.append(dat_bbox[dat_bbox['t']==unique_time])

            time_upperbound = end_time
        sampled_bboxes = np.concatenate(sampled_bboxes)
        mmp = np.lib.format.open_memmap(new_bbox_file, "w+", dtype, sampled_bboxes.shape)
        mmp[:] = sampled_bboxes[:]
        mmp.flush()
        pbar.update(1)
    pbar.close()