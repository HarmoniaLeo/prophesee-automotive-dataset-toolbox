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
        img_ecd[:,:1] = torch.where(forward, torch.cat([torch.zeros_like(forward).float(),torch.zeros_like(forward).float() -1e8],dim=3), img_ecd[:,:1])
    else:
        ecd = torch.where(forward, torch.zeros_like(forward).float() -1e8, torch.zeros_like(forward).float())   #ecd: hw, 1, 2, 1
        img_ecd = torch.cat([img, torch.cat([ecd,ecd],dim=1)],dim=3)    #img_ecd: hw, 2, 2, 2
    if img_ecd.shape[1] > volume_bins:
        img_ecd = img_ecd[:,1:]
    torch.cuda.synchronize()
    generate_encode_time = time.time() - tick
    #print("generate encode",time.time() - tick)

    img_ecd_viewed = img_ecd.view((H, W, img_ecd.shape[1] * 2, 2)).permute(2, 0, 1, 3)
    return img_ecd_viewed, img_ecd, generate_volume_time, generate_encode_time

def generate_taf_cuda(events, shape, past_volume, volume_bins, bin_start, bin_end, infer_start, infer_end):
    x, y, t, p, z = events.unbind(-1)

    x, y, p = x.long(), y.long(), p.long()
    
    histogram_ecd, past_volume, generate_volume_time, generate_encode_time = taf_cuda(x, y, t, p, shape, volume_bins, past_volume)

    return histogram_ecd, past_volume, generate_volume_time, generate_encode_time

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = np.nonzero(dense_tensor)

    features = dense_tensor[non_zero_indices[0],non_zero_indices[1],non_zero_indices[2],non_zero_indices[3]]

    return np.stack(non_zero_indices), features
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)
    parser.add_argument('-target_dir', type=str)
    parser.add_argument('-events_window_abin', type=int, default=10000)
    parser.add_argument('-dataset', type=str, default="gen4")

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

    events_window_abin = args.events_window_abin
    infer_time = 10000
    event_volume_bins = 5
    events_window = events_window_abin * event_volume_bins
    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]
    #raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    #target_dir = "/data/lbd/ATIS_taf"
    #raw_dir = "/data/Large_Automotive_Detection_Dataset_sampling"
    #target_dir = "/data/Large_taf"

    ecd_types = ["quantile","quantile2","minmax","minmax2","leaky"]
    

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
            if not file_name == "17-04-13_15-05-43_3599500000_3659500000":
                continue
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
            
            volume = torch.zeros([event_volume_bins, shape[0], shape[1], 2]).cuda()
            volume[...,1] = -1e8

            time_iter = 0
            bin_start = 0

            while f_event.done:
                events_ = f_event.load_delta_t(infer_time)

                volume = generate_taf_cuda(events_, shape, volume, event_volume_bins, bin_start, bin_start + events_window_abin, time_iter, time_iter + infer_time)

                if np.sum((unique_ts <= time_iter + infer_time)&(unique_ts > time_iter)) > 0:
                    save_volume(volume.cpu().numpy().copy(), unique_ts[(unique_ts <= time_iter + infer_time)&(unique_ts > time_iter)], target_shape)

                torch.cuda.empty_cache()

                time_iter += infer_time
                if time_iter >= bin_start + events_window_abin:
                    bin_start += events_window_abin
            #h5.close()
            pbar.update(1)
        pbar.close()