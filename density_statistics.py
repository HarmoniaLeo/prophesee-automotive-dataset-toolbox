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

events_window_abin = 10000
event_volume_bins = 1
events_window = events_window_abin * event_volume_bins
#shape = [720,1280]
target_shape = [320, 640]
shape = [240,304]
raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
#raw_dir = "/data/Large_Automotive_Detection_Dataset_sampling"
# target_dir = "/data/Large_taf"

for mode in ["train","val","test"]:
    
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

    file_names = []
    time_stamps = []
    densitys = []
    densitys_n = []
    densitys_p = []
    densitys_eff_max = []
    densitys_eff_min = []
    car_counts = 0
    per_counts = 0
    small_counts = 0
    medium_counts = 0
    large_counts = 0
    densitys_bounding_boxes = []
    densitys_bounding_boxes_small = []
    densitys_bounding_boxes_medium = []
    densitys_bounding_boxes_large = []
    densitys_bounding_boxes_cars = []
    densitys_bounding_boxes_pers = []
    pers_count = []
    cars_count = []
    smalls_count = []
    larges_count = []
    mediums_count = []

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


        for bbox_count,unique_time in enumerate(unique_ts):
            end_time = int(unique_time)
            start_time = end_time - events_window

            dat_event = f_event
            if start_time >=0:
                dat_event.seek_time(start_time)
                events = dat_event.load_delta_t(int(end_time - start_time))
            else:
                dat_event.seek_time(0)
                events = dat_event.load_delta_t(int(end_time))
            del dat_event
            events = torch.from_numpy(rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)).cuda()

            density = len(events)/(shape[0]*shape[1])
            density_p = len(events[events[:,3]==1])/(shape[0]*shape[1])
            density_n = len(events[events[:,3]==0])/(shape[0]*shape[1])
            car_count =0 
            per_count = 0
            small_count =0 
            medium_count =0 
            large_count = 0
            max_density = 0
            min_density = np.inf
            gt_trans = dat_bbox[dat_bbox['t'] == unique_time]
            for j in range(len(gt_trans)):
                x, y, w, h = gt_trans['x'][j], gt_trans['y'][j], gt_trans['w'][j], gt_trans['h'][j]
                area = w * h
                points = len(events[(events[:,0]>x)&(events[:,0]<x+w)&(events[:,1]>y)&(events[:,1]<y+h)])
                densitys_bounding_boxes.append(points / (w+h)/2)
                if area < 32*32:
                    small_counts+=1
                    small_count +=1
                    densitys_bounding_boxes_small.append(points / (w*h))
                elif area < 96*96:
                    medium_counts+=1
                    medium_count +=1
                    densitys_bounding_boxes_medium.append(points / (w*h))
                else:
                    densitys_bounding_boxes_large.append(points / (w*h))
                    large_counts+=1
                    large_count +=1
                if points / (shape[0]+shape[1])/2 > max_density:
                    max_density = points / (w+h)/2
                if points / area < min_density:
                    min_density = points / (w+h)/2
                if gt_trans[j][5] == 0:
                    car_counts += 1
                    car_count += 1
                    densitys_bounding_boxes_cars.append(points / (w*h))
                else:
                    per_counts += 1
                    per_count += 1
                    densitys_bounding_boxes_pers.append(points / (w*h))
            
            file_names.append(file_name)
            time_stamps.append(unique_time)
            densitys.append(density)
            densitys_n.append(density_n)
            densitys_p.append(density_p)
            densitys_eff_max.append(max_density)
            densitys_eff_min.append(min_density)
            pers_count.append(per_count)
            cars_count.append(car_count)
            larges_count.append(large_count)
            mediums_count.append(medium_count)
            smalls_count.append(small_count)

        #h5.close()
        pbar.update(1)
    print(car_counts, per_counts, small_counts, medium_counts, large_counts)
    pbar.close()
    csv_path = os.path.join(file_dir,"density_"+mode+".csv")
    pd.DataFrame({
        "File name":file_names,
        "Time stamp":time_stamps,
        "Density":densitys,
        "Density negative":densitys_n,
        "Density positive":densitys_p,
        "Density effective max":densitys_eff_max,
        "Density effective min":densitys_eff_min,
        "Pers count":pers_count,
        "Cars count":cars_count,
        "Larges count":larges_count,
        "Mediums count":mediums_count,
        "Smalls count":smalls_count}).to_csv(csv_path)
    csv_path = os.path.join(file_dir,"density_boxes_"+mode+".csv")
    pd.DataFrame({
        "Density":densitys_bounding_boxes,
        "Density small":densitys_bounding_boxes_small,
        "Density large":densitys_bounding_boxes_large,
        "Density medium":densitys_bounding_boxes_medium,
        "Density car":densitys_bounding_boxes_cars,
        "Density per":densitys_bounding_boxes_pers,}).to_csv(csv_path)