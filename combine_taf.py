from itertools import count
import numpy as np
from sklearn import datasets
from sqlalchemy import false
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import h5py
import pickle
import torch
import time
import math
import argparse
import torch.nn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)
    parser.add_argument('-target_dir', type=str)
    parser.add_argument('-dataset', type=str, default="gen4")
    parser.add_argument('-filter', type=bool, default=False)

    args = parser.parse_args()
    raw_dir = args.raw_dir
    target_dir = args.target_dir
    dataset = args.dataset

    min_event_count = 50000000
    if dataset == "gen4":
        # min_event_count = 800000
        shape = [720,1280]
        target_shape = [512, 640]
    elif dataset == "kitti":
        # min_event_count = 800000
        shape = [375,1242]
        target_shape = [192, 640]
    else:
        # min_event_count = 200000
        shape = [240,304]
        target_shape = [256, 320]
    events_window_abin = 10000
    event_volume_bins = 8
    events_window = events_window_abin * event_volume_bins

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    #bins_saved = [1, 4, 8]
    #transform_applied = [90]
    #transform_applied = [10, 25, 50, 75, 100, 200, "quantile"]

    for mode in ["train","val","test"]:
        file_dir = os.path.join(raw_dir, mode)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        root = file_dir
        target_root = os.path.join(target_dir, mode)
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        #h5 = h5py.File(raw_dir + '/ATIS_taf_'+mode+'.h5', 'w')
        try:
            files = os.listdir(file_dir)
        except Exception:
            continue
        # Remove duplicates (.npy and .dat)
        # files = files[int(2*len(files)/3):]
        #files = files[int(len(files)/3):]
        files = [time_seq_name[:-9] for time_seq_name in files
                        if time_seq_name[-3:] == 'npy']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(files):
            # if not file_name == "17-04-13_15-05-43_3599500000_3659500000":
            #     continue
            # if not file_name == "moorea_2019-06-26_test_02_000_976500000_1036500000":
            #     continue
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(root, file_name + '_bbox.npy')
            #h5 = h5py.File(volume_save_path, "w")
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            time_upperbound = -1e16
            count_upperbound = -1
            already = False
            sampling = False

            #min_event_count = f_event.event_count()

            for bbox_count,unique_time in enumerate(unique_ts):
                ecds = []
                try:
                    for i in range(event_volume_bins):
                        ecd_file = os.path.join(os.path.join(target_root,"bin{0}".format(7-i)), file_name+ "_" + str(unique_time) + ".npy")
                        ecd = np.fromfile(ecd_file, dtype=np.uint8).reshape(2, target_shape[0], target_shape[1]).astype(np.float32)
                        ecds.append(ecd)
                except Exception:
                    continue
                #volume = np.concatenate([feature, ecd], 0)
                volume = np.concatenate(ecds, 0)
                            
                if not os.path.exists(os.path.join(target_root,"bins{0}".format(event_volume_bins))):
                    os.makedirs(os.path.join(target_root,"bins{0}".format(event_volume_bins)))
                volume.astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"bins{0}".format(event_volume_bins)),file_name+"_"+str(unique_time)+".npy"))

            #h5.close()
            pbar.update(1)
        pbar.close()
        # if mode == "test":
        #     np.save(os.path.join(root, 'total_volume_time.npy'),np.array(total_volume_time))
        #     np.save(os.path.join(root, 'total_taf_time.npy'),np.array(total_taf_time))
        #h5.close()