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

def generate_taf_cuda(events, shape, past_volume = None, volume_bins=5):
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
    events_window_abin = 10000
    event_volume_bins = 5
    events_window = events_window_abin * event_volume_bins
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
                # if unique_time <= 500000:
                #     continue
                # if (not sampling) and (unique_time - time_upperbound < 450000):
                #     continue
                # else:
                #     if not sampling:
                #         sampling_start_time = unique_time
                #         sampling = True
                #     if unique_time - sampling_start_time > 50000:
                #         sampling = False
                #         continue
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_count = end_count - min_event_count
                print(start_count, count_upperbound)
                if start_count < 0:
                    start_count = 0
                f_event.seek_event(start_count)
                start_time = int(f_event.current_time)
                if (end_time - start_time) < events_window:
                    start_time = end_time - events_window
                else:
                    start_time = end_time - round((end_time - start_time - events_window)/events_window_abin) * events_window_abin - events_window

                #assert (start_time < time_upperbound) or (time_upperbound < 0)
                if start_time > time_upperbound:
                    start_count = f_event.seek_time(start_time)
                    if (start_count is None) or (start_time < 0):
                        start_count = 0
                    memory = None
                else:
                    start_count = count_upperbound
                    start_time = time_upperbound
                    end_time = round((end_time - start_time) / events_window_abin) * events_window_abin + start_time
                    if end_time > f_event.total_time():
                        end_time = f_event.total_time()
                    end_count = f_event.seek_time(end_time)
                    assert bbox_count > 0

                
                #if not (os.path.exists(volume_save_path)):
                dat_event = f_event
                dat_event.seek_event(start_count)

                events = dat_event.load_n_events(int(end_count - start_count))
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

                if start_time > time_upperbound:
                    memory = None
                #     events_ = events[events[...,4] < event_volume_bins]
                #     t_max = start_time + event_volume_bins * events_window_abin
                #     t_min = start_time
                #     events_[:,2] = (events_[:, 2] - t_min)/(t_max - t_min + 1e-8)
                #     volume, memory, generate_volume_time, generate_encode_time = generate_taf_cuda(events_, target_shape, memory, event_volume_bins)
                #     iter = event_volume_bins
                # else:
                for iter in range(bins):
                    events_ = events[events[...,4] == iter]
                    t_max = start_time + (iter + 1) * events_window_abin
                    t_min = start_time + iter * events_window_abin
                    events_[:,2] = (events_[:, 2] - t_min)/(t_max - t_min + 1e-8)
                    #tick = time.time()
                    volume, memory, generate_volume_time, generate_encode_time = generate_taf_cuda(events_, target_shape, memory, event_volume_bins)
                    #print(generate_volume_time, generate_encode_time)
                    #torch.cuda.synchronize()
                    if mode == "test":
                        total_volume_time.append(generate_volume_time)
                        total_taf_time.append(generate_volume_time + generate_encode_time)
                    #print(total_time/generate_times)
                volume_ = volume.cpu().numpy().copy()
                volume_[...,1] = np.where(volume_[...,1]>-1e8, volume_[...,1] - 1, 0)
                locations, features = denseToSparse(volume_)
                c, y, x, p = locations
                
                locations = x.astype(np.uint32) + np.left_shift(y.astype(np.uint32), 10) + np.left_shift(c.astype(np.uint32), 19) + np.left_shift(p.astype(np.uint32), 23)
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