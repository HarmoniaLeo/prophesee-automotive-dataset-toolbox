from cgitb import small
import numpy as np
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import pandas as pd
import torch
import time
import math
import argparse
import cv2

def compute_TVL1(prev, curr, bound=1):
    """Compute the TV-L1 optical flow."""
    TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
    #TVL1 = cv2.DualTVL1OpticalFlow_create()
    #TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    # assert flow.dtype == np.float32
    
    # flow = np.sqrt(flow[:,:,:1] ** 2 + flow[:,:,1:2] ** 2)
    # flow = (flow + bound) * (255.0 / (2 * bound))
    # flow = np.round(flow).astype(int)
    # flow[flow >= 255] = 255
    # flow[flow <= 0] = 0
 
    return flow

def cal_for_frames(volume1, volume2):
 
    prev = volume1
    #prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr = volume2
    flow = compute_TVL1(prev, curr)
 
    return flow
 
 
def extract_flow(volume1, volume2):
    flow = cal_for_frames(volume1, volume2)
    return flow

def generate_timesurface(events,shape,end_stamp,buffer):
    if not (buffer is None):
        volume1 = buffer
        volume2 = buffer
    else:
        volume1, volume2 = np.zeros(shape), np.zeros(shape)
    # end_stamp = events[:,2].max()
    # start_stamp = events[:,2].min()
    for event in events:
        if event[2] < end_stamp - 50000:
            volume1[int(event[1])][int(event[0])] = event[2]
        volume2[int(event[1])][int(event[0])] = event[2]
    buffer = volume2
    volume2 = volume2 - 50000
    volume1 = (volume1 - np.min(np.min(volume1))) / (np.max(np.max(volume1)) - np.min(np.min(volume1))) * 255
    volume2 = (volume2 - np.min(np.min(volume2))) / (np.max(np.max(volume2)) - np.min(np.min(volume2))) * 255
    # volume1 = volume1 - events[:,2].max() + 50000
    # volume2 = volume2 - events[:,2].max() + 40000
    # volume1 = volume1 / 50000 * 255
    # volume2 = volume2 / 50000 * 255
    # volume1 = np.where(volume1<0, 0, volume1)
    # volume2 = np.where(volume2<0, 0, volume2)
    return volume1.astype(np.uint8), volume2.astype(np.uint8), buffer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-dataset', type=str)

    args = parser.parse_args()
    mode = "test"

    if args.dataset == "gen1":
        raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
        shape = [240,304]
        events_window_abin = 800000
    else:
        raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
        shape = [720,1280]
        events_window_abin = 10000000
    
    result_path = "statistics_result"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
        
    file_dir = os.path.join(raw_dir, mode)
    root = file_dir
    #h5 = h5py.File(raw_dir + '/ATIS_taf_'+mode+'.h5', 'w')
    files = os.listdir(file_dir)

    # Remove duplicates (.npy and .dat)
    files = [time_seq_name[:-7] for time_seq_name in files
                    if time_seq_name[-3:] == 'dat']

    pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

    file_names = []
    gt = []
    densitys = []

    for i_file, file_name in enumerate(files):
        # if i_file>5:
        #     break
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
        time_surface_buffer = None

        for bbox_count,unique_time in enumerate(unique_ts):
            end_time = int(unique_time)

            current_event = f_event.seek_time(end_time)

            start_event = current_event - events_window_abin

            f_event.seek_event(start_event)
            start_time = f_event.current_time

            dat_event = f_event
            if start_time > time_upperbound:
                dat_event.seek_time(start_time)
                time_surface_buffer = None
            else:
                dat_event.seek_time(time_upperbound)
                start_time = time_upperbound

            events = dat_event.load_delta_t(int(end_time-start_time))
            events = rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)

            del dat_event

            gt_trans = dat_bbox[dat_bbox['t'] == unique_time]

            volume1, volume2, time_surface_buffer = generate_timesurface(events, shape, end_time, time_surface_buffer)
            flow = extract_flow(volume1, volume2)

            for j in range(len(gt_trans)):
                x, y, w, h = gt_trans['x'][j], gt_trans['y'][j], gt_trans['w'][j], gt_trans['h'][j]
                file_names.append(file_name)
                gt.append(gt_trans[j])

                density = np.sum(np.sqrt(flow[int(x):int(x+w),int(y):int(y+h)]**2)/w/h)
                densitys.append(density)

        #h5.close()
        pbar.update(1)
    pbar.close()
    csv_path = os.path.join(result_path,"gt_"+args.dataset+".npz")
    print([np.percentile(densitys,q/100) for q in range(0,100,5)])
    np.savez(csv_path,
        file_names = file_names,
        gts = gt,
        densitys = densitys)