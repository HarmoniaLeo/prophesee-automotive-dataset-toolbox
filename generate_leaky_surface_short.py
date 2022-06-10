from itertools import count
import numpy as np
from sklearn import datasets
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


def generate_leakysurface(events,shape,memory,lamda):
    if memory is None:
        q, p = np.zeros(shape), np.zeros(shape)
    else:
        q, p = memory
    t_prev = 0
    for event in events:
        if event[3] == 1:
            delta = event[2] - t_prev
            q = np.where(p - lamda * delta < 0, 0, p - lamda * delta)
            p = q
            p[event[1]][event[0]] += 1
        t_prev = event[2]
    return q, (q, p)

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = np.nonzero(dense_tensor)

    features = dense_tensor[non_zero_indices[0],non_zero_indices[1]]

    return np.stack(non_zero_indices), features

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)
    parser.add_argument('-target_dir', type=str)
    parser.add_argument('-dataset', type=str, default="gen4")

    args = parser.parse_args()
    raw_dir = args.raw_dir
    target_dir = args.target_dir
    dataset = args.dataset

    min_event_count = 2400000
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
    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]
    #raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    #target_dir = "/data/lbd/ATIS_taf"
    #raw_dir = "/data/Large_Automotive_Detection_Dataset_sampling"
    #target_dir = "/data/Large_taf"
    

    total_volume_time = []
    total_taf_time = []

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

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
        files = [time_seq_name[:-7] for time_seq_name in files
                        if time_seq_name[-3:] == 'dat']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(files):
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(root, file_name + '_bbox.npy')
            # if os.path.exists(volume_save_path):
            #     continue
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
                volume_save_path_l = os.path.join(target_root, file_name+"_"+str(unique_time)+"_locations.npy")
                volume_save_path_f = os.path.join(target_root, file_name+"_"+str(unique_time)+"_features.npy")
                if os.path.exists(volume_save_path_f) and os.path.exists(volume_save_path_l):
                    continue
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_count = end_count - min_event_count
                if start_count < 0:
                    start_count = 0
                f_event.seek_event(start_count)
                start_time = int(f_event.current_time)

                #assert (start_time < time_upperbound) or (time_upperbound < 0)
                if start_time > time_upperbound:
                    start_count = f_event.seek_time(start_time)
                    if (start_count is None) or (start_time < 0):
                        start_count = 0
                    memory = None
                else:
                    start_count = count_upperbound
                    start_time = time_upperbound
                    assert bbox_count > 0

                
                #if not (os.path.exists(volume_save_path)):
                dat_event = f_event
                dat_event.seek_event(start_count)

                events = dat_event.load_n_events(int(end_count - start_count))
                del dat_event
                #events = torch.from_numpy(rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)).cuda()
                events = rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)
                events[:,0] = events[:,0] * rw
                events[:,1] = events[:,1] * rh

                if start_time > time_upperbound:
                    memory = None

                volume, memory = generate_leakysurface(events, target_shape, memory, 1e-4)

                #volume_ = volume.cpu().numpy().copy()

                locations, features = denseToSparse(volume)
                y, x = locations
                
                locations = x.astype(np.uint32) + np.left_shift(y.astype(np.uint32), 10)
                locations.tofile(volume_save_path_l)
                features.tofile(volume_save_path_f)
                #np.savez(volume_save_path, locations = locations, features = features)
                #h5.create_dataset(str(unique_time)+"/locations", data=locations)
                #h5.create_dataset(str(unique_time)+"/features", data=features)
                time_upperbound = end_time
                count_upperbound = end_count
                torch.cuda.empty_cache()
            #h5.close()
            pbar.update(1)
        pbar.close()
        # if mode == "test":
        #     np.save(os.path.join(root, 'total_volume_time.npy'),np.array(total_volume_time))
        #     np.save(os.path.join(root, 'total_taf_time.npy'),np.array(total_taf_time))
        #h5.close()