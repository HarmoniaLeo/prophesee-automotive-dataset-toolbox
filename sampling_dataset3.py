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


events_window_total = 50000
raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
target_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling2"

for mode in ["train","val","test"]:
    
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
        event_file = os.path.join(root, file_name + '_td.dat')
        bbox_file = os.path.join(root, file_name + '_bbox.npy')
        new_event_file = os.path.join(target_root, file_name + '_td.dat')
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
        f_event = psee_loader.PSEELoader(event_file)

        f_event_new = open(new_event_file, "wb")

        already = False
        sampling = False

        sampled_events = []
        sampled_bboxes = []
        for bbox_count,unique_time in enumerate(unique_ts):
        
            sampling_start_time = unique_time - events_window_total
            end_time = int(unique_time)
            f_event.seek_time(sampling_start_time)
            events = f_event.load_delta_t(events_window_total)

            sampled_events.append(events)
            sampled_bboxes.append(dat_bbox[dat_bbox['t']==unique_time])

        dat_events_tools.write_event_buffer(f_event_new, np.concatenate(sampled_events))
        sampled_bboxes = np.concatenate(sampled_bboxes)
        mmp = np.lib.format.open_memmap(new_bbox_file, "w+", dtype, sampled_bboxes.shape)
        mmp[:] = sampled_bboxes[:]
        mmp.flush()
        pbar.update(1)
    pbar.close()