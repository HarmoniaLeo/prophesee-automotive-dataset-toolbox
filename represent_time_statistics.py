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

def taf_cuda(x, y, t, p, shape, volume_bins, past_volume):
    tick = time.time()
    H, W = shape

    t_star = t.float()[:,None,None]
    
    adder = torch.stack([torch.arange(2),torch.arange(2)],dim = 1).to(x.device)[None,:,:]   #1, 2, 2
    adder = (1 - torch.abs(adder-t_star)) * torch.stack([p,1 - p],dim=1)[:,None,:]  #n, 2, 2
    adder = torch.where(adder>=0,adder,torch.zeros_like(adder)).view(adder.shape[0], 4) #n, 4
    
    img = torch.zeros((H * W, 4)).float().to(x.device)
    img.index_add_(0, x + W * y, adder)

    img = img.view(H * W, 2, 2, 1) #img: hw, 2, 2, 1
    torch.cuda.synchronize()
    generate_volume_time = time.time() - tick
    #print("generate_volume_time",time.time() - tick)

    tick = time.time()
    forward = (img[:,-1]==0)[:,None]   #forward: hw, 1, 2, 1
    if not (past_volume is None):
        img_old_ecd = past_volume    #img_ecd: hw, 2, 2, 2
        img_old_ecd[:,-1,:,0] = torch.where(img_old_ecd[:,-1,:,1] == 0,img_old_ecd[:,-1,:,0] + img[:,0,:,0],img_old_ecd[:,-1,:,0])
        img_ecd = torch.cat([img_old_ecd,torch.cat([img[:,1:],torch.zeros_like(img[:,1:])],dim=3)],dim=1)
        for i in range(1,img_ecd.shape[1])[::-1]:
            img_ecd[:,i-1,:,1] = img_ecd[:,i-1,:,1] - 1
            img_ecd[:,i:i+1] = torch.where(forward, img_ecd[:,i-1:i],img_ecd[:,i:i+1])
        img_ecd[:,:1] = torch.where(forward, torch.cat([torch.zeros_like(forward).float(),torch.zeros_like(forward).float() -1e6],dim=3), img_ecd[:,:1])
    else:
        ecd = torch.where(forward, torch.zeros_like(forward).float() -1e6, torch.zeros_like(forward).float())   #ecd: hw, 1, 2, 1
        img_ecd = torch.cat([img, torch.cat([ecd,ecd],dim=1)],dim=3)    #img_ecd: hw, 2, 2, 2
    if img_ecd.shape[1] > volume_bins:
        img_ecd = img_ecd[:,1:]
    torch.cuda.synchronize()
    generate_encode_time = time.time() - tick
    #print("generate encode",time.time() - tick)

    img_ecd_viewed = img_ecd.view((H, W, img_ecd.shape[1] * 2, 2)).permute(2, 0, 1, 3)
    return img_ecd_viewed, img_ecd, generate_volume_time, generate_encode_time

def generate_taf_cuda(events, shape, past_volume = None, volume_bins=5):
    x, y, t, p, z = events.unbind(-1)

    x, y, p = x.long(), y.long(), p.long()
    
    histogram_ecd, past_volume, generate_volume_time, generate_encode_time = taf_cuda(x, y, t, p, shape, volume_bins, past_volume)

    return histogram_ecd, past_volume, generate_volume_time, generate_encode_time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-dataset', type=str)
    args = parser.parse_args()

    events_window_abin = 10000
    event_volume_bins = 5
    events_window = events_window_abin * (event_volume_bins + 1)
    
    if args.dataset == "gen4":
        shape = [720,1280]
        target_shape = [512, 640]
        raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
    else:
        shape = [240,304]
        target_shape = [256, 320]
        raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]
    #shape = [240,304]
    #raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    # target_dir = "/data/Large_taf"

    for mode in ["test"]:
        
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
        generate_volume_times = []
        generate_taf_times = []

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

            history_density = -1

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

                events[:,0] = events[:,0] * rw
                events[:,1] = events[:,1] * rh

                z = torch.zeros_like(events[:,0])

                bins = math.ceil((end_time - start_time) / events_window_abin)
                
                for i in range(bins):
                    z = torch.where((events[:,2] >= start_time + i * events_window_abin)&(events[:,2] <= start_time + (i + 1) * events_window_abin), torch.zeros_like(events[:,2])+i, z)
                    #events_timestamps.append(start_time + (i + 1) * self.events_window_abin)
                events = torch.cat([events,z[:,None]], dim=1)

                memory = None
                for iter in range(bins):
                    events_ = events[events[...,4] == iter]
                    t_max = start_time + (iter + 1) * events_window_abin
                    t_min = start_time + iter * events_window_abin
                    events_[:,2] = (events_[:, 2] - t_min)/(t_max - t_min + 1e-8)
                    volume, memory, generate_volume_time, generate_encode_time = generate_taf_cuda(events_, target_shape, memory, event_volume_bins)
                
                file_names.append(file_name)
                time_stamps.append(unique_time)
                generate_volume_times.append(generate_volume_time)
                generate_taf_times.append(generate_volume_time + generate_encode_time)

            #h5.close()
            pbar.update(1)
        pbar.close()
        print("Generate volume time: ",np.mean(generate_volume_times))
        print("Generate taf time: ",np.mean(generate_taf_times))
        # csv_path = os.path.join(file_dir,"generate_time.csv")
        # pd.DataFrame({
        #     "File name":file_names,
        #     "Time stamp":time_stamps,
        #     "Generate volume time":generate_volume_times,
        #     "Generate taf time":generate_taf_times}).to_csv(csv_path)