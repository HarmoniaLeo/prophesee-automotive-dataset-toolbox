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
import cv2

def generate_taf_cuda(x, y, c, p, features, volume_bins, shape):

    x, y, p, c = x.long(), y.long(), p.long(), c.long()
    
    H, W = shape
    C = volume_bins * 2

    feature_map = torch.zeros(C * H * W * 2).float().to(x.device)
    feature_map.index_add_(0, c * H * W * 2 + y * W * 2 + x * 2 + p, features.float())

    volume = feature_map.view(C, H, W, 2).contiguous()
    volume[:,:,:,1] = torch.where(volume[:,:,:,1] ==0, torch.zeros_like(volume[:,:,:,1]).float() - 1e8, volume[:,:,:,1] + 1)

    features = volume[:,:,:,0] / 5 * 255

    ecd_quantile = volume[:,:,:,1].clone()

    for i in range(len(features)):

        ecd_view = volume[i,:,:,1] [volume[i,:,:,1]  > - 1e8]
        
        try:
            q10, q90 = torch.quantile(ecd_view, torch.tensor([0.1,0.9]).to(x.device))
            q100 = torch.max(ecd_view)
            ecd_quantile[i] = torch.where(ecd_quantile[i] > q90, (ecd_quantile[i] - q90) / (q100 - q90 + 1e-8) * 2, ecd_quantile[i])
            ecd_quantile[i] = torch.where((ecd_quantile[i] <= q90)&(ecd_quantile[i] > - 1e8), (ecd_quantile[i] - q90) / (q90 - q10 + 1e-8) * 6, ecd_quantile[i])
            ecd_quantile[i] = torch.exp(ecd_quantile[i]) / 7.389 * 255
        except Exception:
            pass
    
    ecd_quantile2 = volume[:,:,:,1].clone()
    ecd_view = volume[:,:,:,1] [volume[:,:,:,1]  > - 1e8]
    try:
        q10, q90 = torch.quantile(ecd_view, torch.tensor([0.1,0.9]).to(x.device))
        q100 = torch.max(ecd_view)
        ecd_quantile2 = torch.where(ecd_quantile2 > q90, (ecd_quantile2 - q90) / (q100 - q90 + 1e-8) * 2, ecd_quantile2)
        ecd_quantile2 = torch.where((ecd_quantile2 <= q90)&(ecd_quantile2 > - 1e8), (ecd_quantile2 - q90) / (q90 - q10 + 1e-8) * 6, ecd_quantile2)
        ecd_quantile2 = torch.exp(ecd_quantile2) / 7.389 * 255
    except Exception:
        pass
    
    ecd_quantile3 = volume[:,:,:,1].clone()
    ecd_view = volume[:,:,:,1] [volume[:,:,:,1]  > - 1e8]
    try:
        q10, q90 = torch.quantile(ecd_view, torch.tensor([0.1,0.95]).to(x.device))
        q100 = torch.max(ecd_view)
        ecd_quantile2 = torch.where(ecd_quantile2 > q90, (ecd_quantile2 - q90) / (q100 - q90 + 1e-8) * 2, ecd_quantile2)
        ecd_quantile2 = torch.where((ecd_quantile2 <= q90)&(ecd_quantile2 > - 1e8), (ecd_quantile2 - q90) / (q90 - q10 + 1e-8) * 6, ecd_quantile2)
        ecd_quantile2 = torch.exp(ecd_quantile2) / 7.389 * 255
    except Exception:
        pass
        
    ecd_minmax = volume[:,:,:,1].clone()
    ecd_view = volume[:,:,:,1] [volume[:,:,:,1]  > - 1e8]
    try:
        q100 = torch.max(ecd_view)
        q0 = torch.min(ecd_view)
        ecd_minmax = torch.where(ecd_minmax > -1e8, (ecd_minmax - q100) / (q100 - q0 + 1e-8) * 6, ecd_minmax)
        ecd_minmax = torch.exp(ecd_minmax) * 255
    except Exception:
        pass

    ecd_leaky = volume[:,:,:,1].clone() * 0.1
    ecd_leaky = torch.exp(ecd_leaky) * 255

    features = torch.where(features > 255, torch.zeros_like(features)+ 255, features)
    ecd_quantile = torch.where(ecd_quantile > 255, torch.zeros_like(ecd_quantile)+ 255, ecd_quantile)
    ecd_quantile2 = torch.where(ecd_quantile2 > 255, torch.zeros_like(ecd_quantile2)+ 255, ecd_quantile2)
    ecd_minmax = torch.where(ecd_minmax > 255, torch.zeros_like(ecd_minmax)+ 255, ecd_minmax)
    ecd_leaky = torch.where(ecd_leaky > 255, torch.zeros_like(ecd_leaky)+ 255, ecd_leaky)

    return features, [ecd_quantile, ecd_quantile2, ecd_quantile3, ecd_minmax, ecd_leaky]


def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = np.nonzero(dense_tensor)

    features = dense_tensor[non_zero_indices[0],non_zero_indices[1],non_zero_indices[2]]

    return np.stack(non_zero_indices), features
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='visualize one or several event files along with their boxes')
    parser.add_argument('-bbox_dir', type=str)
    parser.add_argument('-raw_dir', type=str)
    parser.add_argument('-target_dir', type=str)
    parser.add_argument('-dataset', type=str, default="gen4")

    args = parser.parse_args()
    bbox_dir = args.bbox_dir
    raw_dir = args.raw_dir
    target_dir = args.target_dir
    dataset = args.dataset

    if dataset == "gen4":
        # min_event_count = 800000
        shape = [512, 640]
        target_shape = [512, 640]
    else:
        # min_event_count = 200000
        shape = [240, 304]
        target_shape = [256, 320]
    event_volume_bins = 5
    rh = shape[0] / target_shape[0]
    rw = shape[1] / target_shape[1]

    ecd_types = ["quantile","quantile2","quantile3","minmax","leaky"]

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    #for mode in ["train","val","test"]:
    for mode in ["train"]:
    #for mode in ["test"]:
        file_dir = os.path.join(bbox_dir, mode)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        root = file_dir
        data_dir = os.path.join(raw_dir, mode)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        data_root = data_dir
        target_root = os.path.join(target_dir, mode)
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        for ecd_type in ["feature"] + ecd_types:
            if not os.path.exists(os.path.join(target_root,ecd_type)):
                os.makedirs(os.path.join(target_root,ecd_type))
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
            if not file_name == "moorea_2019-06-26_test_02_000_976500000_1036500000":
                continue
            bbox_file = os.path.join(root, file_name + '_bbox.npy')
            # if os.path.exists(volume_save_path):
            #     continue
            #h5 = h5py.File(volume_save_path, "w")
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            #min_event_count = f_event.event_count()

            for bbox_count,unique_time in enumerate(unique_ts):
                volume_save_path_l = os.path.join(data_root, file_name+"_"+str(unique_time)+"_locations.npy")
                volume_save_path_f = os.path.join(data_root, file_name+"_"+str(unique_time)+"_features.npy")
                end_time = int(unique_time)
                # end_count = f_event.seek_time(end_time)
                # if end_count is None:
                #     continue
                if os.path.exists(volume_save_path_l):
                    locations = np.fromfile(volume_save_path_l, dtype=np.int32)
                    x = np.bitwise_and(locations, 1023).astype(np.float32)
                    y = np.right_shift(np.bitwise_and(locations, 523264), 10).astype(np.float32)
                    c = np.right_shift(np.bitwise_and(locations, 7864320), 19).astype(np.float32)
                    p = np.right_shift(np.bitwise_and(locations, 8388608), 23).astype(np.float32)

                    features = np.fromfile(volume_save_path_f, dtype=np.float32)

                    x = x * rw
                    y = y * rh

                    if target_shape[0] != shape[0]:
                        x = np.where(x%19!=0, x+1, x)
                        y = np.where(y%15!=0, y+1, y)

                    features, ecds = generate_taf_cuda(torch.from_numpy(x).cuda(),torch.from_numpy(y).cuda(),torch.from_numpy(c).cuda(),torch.from_numpy(p).cuda(),torch.from_numpy(features).cuda(),event_volume_bins,shape)
                    if target_shape[0] != shape[0]:
                        features = torch.nn.functional.interpolate(features[None,:,:,:], size = target_shape, mode='nearest')[0]
                        for i in range(len(ecds)):
                            ecds[i] = torch.nn.functional.interpolate(ecds[i][None,:,:,:], size = target_shape, mode='nearest')[0]
                    features = features.cpu().numpy()
                    ecds = [ecd.cpu().numpy() for ecd in ecds]

                    for i, ecd_type in enumerate(ecd_types):
                        locations, ecd = denseToSparse(ecds[i])

                        c, y, x = locations
                        p = c%2
                        c = (c/2).astype(int)

                        volume = x.astype(np.uint32) + np.left_shift(y.astype(np.uint32), 10) + np.left_shift(c.astype(np.uint32), 19) + np.left_shift(p.astype(np.uint32), 22) + np.left_shift(ecd.astype(np.uint32), 23)

                        x = np.bitwise_and(volume, 1023).astype(int)
                        y = np.right_shift(np.bitwise_and(volume, 523264), 10).astype(int)
                        c = np.right_shift(np.bitwise_and(volume, 3670016), 19).astype(int)
                        p = np.right_shift(np.bitwise_and(volume, 4194304), 22).astype(int)
                        ecd = np.right_shift(np.bitwise_and(volume, 2139095040), 23).astype(int)

                        volume.tofile(os.path.join(os.path.join(target_root,ecd_type),file_name+"_"+str(unique_time)+".npy"))
                    
                    features.astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"feature"),file_name+"_"+str(unique_time)+".npy"))
                    # features.astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"feature"),file_name+"_"+str(unique_time)+".npy"))
                    # ecd_quantile.astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"quantile"),file_name+"_"+str(unique_time)+".npy"))
                    # ecd_minmax.astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"minmax"),file_name+"_"+str(unique_time)+".npy"))
                    # ecd_leaky.astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"leaky"),file_name+"_"+str(unique_time)+".npy"))
            pbar.update(1)
        pbar.close()