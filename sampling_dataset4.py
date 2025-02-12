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


min_event_count = 800000
events_window_abin = 10000
event_volume_bins = 5
events_window = events_window_abin * event_volume_bins
events_window_total = int(50000 + 16667 * 17)
raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset"
target_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"

files_skip = [
    "moorea_2019-06-17_test_02_000_1037500000_1097500000",
    "moorea_2019-06-14_002_1220500000_1280500000",
    "moorea_2019-06-21_000_61500000_121500000",
    "moorea_2019-06-17_test_01_000_732500000_792500000",
    "moorea_2019-06-21_001_2196500000_2256500000",
    "moorea_2019-06-21_000_244500000_304500000",
    "moorea_2019-06-17_test_02_000_244500000_304500000",
    "moorea_2019-06-21_000_427500000_487500000",
    "moorea_2019-06-17_test_02_000_1830500000_1890500000",
    "moorea_2019-06-17_test_02_000_183500000_243500000",
    "moorea_2019-06-17_test_01_000_2135500000_2195500000",
    "moorea_2019-06-19_000_2562500000_2622500000",
    "moorea_2019-06-17_test_02_000_2623500000_2683500000",
    "moorea_2019-06-17_test_02_000_2013500000_2073500000",
    "moorea_2019-06-17_test_02_000_1769500000_1829500000"
]

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
        print(file_name)
        if file_name not in files_skip:
            continue
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

        time_upperbound = -1e16
        count_upperbound = -1
        already = False
        sampling = False

        sampled_events = []
        sampled_bboxes = []
        for bbox_count,unique_time in enumerate(unique_ts):
            if unique_time <= 500000:
                continue
            if (unique_time - time_upperbound < 1000000):
                continue
            else:
                sampling_start_time = unique_time
            end_time = int(unique_time)
            end_count = f_event.seek_time(end_time)
            if end_count is None:
                continue
            start_count = end_count - min_event_count
            if start_count < 0:
                start_count = 0
            f_event.seek_event(start_count)
            start_time = int(f_event.current_time)
            if (end_time - start_time) < events_window_total:
                start_time = end_time - events_window_total
            else:
                start_time = end_time - round((end_time - start_time - events_window)/events_window_abin) * events_window_abin - events_window

            if start_time > time_upperbound:
                start_count = f_event.seek_time(start_time)
                if (start_count is None) or (start_time < 0):
                    start_count = 0
            else:
                start_count = count_upperbound
                start_time = time_upperbound
                end_time = round((end_time - start_time) / events_window_abin) * events_window_abin + start_time
                if end_time > f_event.total_time():
                    end_time = f_event.total_time()
                end_count = f_event.seek_time(end_time)
                assert bbox_count > 0

            
            dat_event = f_event
            dat_event.seek_event(start_count)
            events = dat_event.load_n_events(int(end_count - start_count))
            del dat_event
            sampled_events.append(events)
            sampled_bboxes.append(dat_bbox[dat_bbox['t']==unique_time])

            time_upperbound = end_time
            count_upperbound = end_count
        dat_events_tools.write_event_buffer(f_event_new, np.concatenate(sampled_events))
        sampled_bboxes = np.concatenate(sampled_bboxes)
        mmp = np.lib.format.open_memmap(new_bbox_file, "w+", dtype, sampled_bboxes.shape)
        mmp[:] = sampled_bboxes[:]
        mmp.flush()
        pbar.update(1)
    pbar.close()