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

pooling_layer = torch.nn.MaxPool2d(2, 2)
up_sampling_layer = torch.nn.Upsample(scale_factor=2, mode = "nearest")


def taf_cuda(x, y, t, p, shape, volume_bins, past_volume, filter = False):
    tick = time.time()
    H, W = shape
    
    img = torch.zeros((H * W * 2)).float().to(x.device)
    img.index_add_(0, p + 2 * x + 2 * W * y, torch.ones_like(x).float())

    img = img.view(H, W, 2)
    torch.cuda.synchronize()
    generate_volume_time = time.time() - tick

    tick = time.time()
    if not filter:
        forward = (img == 0)
    else:
        forward = (img <= 1).float()
        forward = 1 - forward.permute(2, 0, 1)[None, :, :, :]
        forward = pooling_layer(forward)
        forward = up_sampling_layer(1 - forward).bool()[0].permute(1, 2, 0)
    torch.cuda.synchronize()
    filter_time = time.time() - tick
    tick = time.time()
    old_ecd = past_volume
    if torch.all(forward):
        ecd = old_ecd
    else:
        ecd = torch.where(forward, torch.zeros_like(forward).float() - 1e8, torch.zeros_like(forward).float())[:, :, :, None]
        ecd = torch.cat([old_ecd, ecd],dim=3)
        for i in range(1,ecd.shape[3])[::-1]:
            ecd[:,:,:,i-1] = ecd[:,:,:,i-1] - 1
            ecd[:,:,:,i] = torch.where(forward, ecd[:,:,:,i-1],ecd[:,:,:,i])
        if ecd.shape[3] > volume_bins:
            ecd = ecd[:,:,:,1:]
        else:
            ecd[:,:,:,0] = torch.where(forward, torch.zeros_like(forward).float() -1e8, ecd[:,:,:,0])
    torch.cuda.synchronize()
    generate_encode_time = time.time() - tick

    ecd_viewed = ecd.permute(3, 2, 0, 1).contiguous().view(volume_bins * 2, H, W)

    #print(generate_volume_time, filter_time, generate_encode_time)
    return ecd_viewed, ecd

def generate_taf_cuda(events, shape, past_volume = None, volume_bins=5, filter = False):
    x, y, t, p, z = events.unbind(-1)

    x, y, p = x.long(), y.long(), p.long()
    
    histogram_ecd, past_volume = taf_cuda(x, y, t, p, shape, volume_bins, past_volume, filter)

    return histogram_ecd, past_volume

def quantile_transform(ecd, head = [90], tail = 10):
    ecd = ecd.clone()
    ecd_view = ecd[ecd > -1e8]
    qs = torch.quantile(ecd_view, torch.tensor([tail] + head).to(ecd_view.device)/100)
    q100 = torch.max(ecd_view)
    q10 = qs[None, None, None, None, 0:1]
    qs = qs[None, None, None, None, 1:]
    ecd = [ecd for i in range(len(head))]
    ecd = torch.stack(ecd, dim = -1)
    ecd = torch.where(ecd > qs, (ecd - qs) / (q100 - qs + 1e-8) * 2, ecd)
    ecd = torch.where((ecd <= qs)&(ecd > - 1e8), (ecd - qs) / (qs - q10 + 1e-8) * 6, ecd)
    ecd = torch.exp(ecd) / 7.389 * 255
    ecd = torch.where(ecd > 255, torch.zeros_like(ecd) + 255, ecd)
    return ecd

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
    event_volume_bins = 10
    events_window = events_window_abin * event_volume_bins

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    bins_saved = [1, 5]
    transform_applied = [90]

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
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_count = end_count - min_event_count
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
                

                z = torch.zeros_like(events[:,0])

                bins = math.ceil((end_time - start_time) / events_window_abin)
                
                for i in range(bins):
                    z = torch.where((events[:,2] >= start_time + i * events_window_abin)&(events[:,2] <= start_time + (i + 1) * events_window_abin), torch.zeros_like(events[:,2])+i, z)
                    #events_timestamps.append(start_time + (i + 1) * self.events_window_abin)
                events = torch.cat([events,z[:,None]], dim=1)

                if start_time > time_upperbound:
                    memory = torch.zeros((shape[0], shape[1], 2, event_volume_bins)).cuda() - 1e8
                for iter in range(bins):
                    events_ = events[events[...,4] == iter]
                    t_max = start_time + (iter + 1) * events_window_abin
                    t_min = start_time + iter * events_window_abin
                    events_[:,2] = (events_[:, 2] - t_min)/(t_max - t_min + 1e-8)
                    #tick = time.time()
                    volume, memory = generate_taf_cuda(events_, shape, memory, event_volume_bins, args.filter)
                    #print(generate_volume_time, generate_encode_time)
                    #torch.cuda.synchronize()
                volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]
                volume = volume.view(event_volume_bins, 2, target_shape[0], target_shape[1])
                for i, bin_saved in enumerate(bins_saved):
                    for j, head in enumerate(transform_applied):
                        ecd = quantile_transform(volume[-bin_saved:], head = [head])
                        ecd = ecd.cpu().numpy().copy()
                        if not os.path.exists(os.path.join(target_root,"quantile{0}_bins{1}".format(head, bin_saved))):
                            os.makedirs(os.path.join(target_root,"quantile{0}_bins{1}".format(head, bin_saved)))
                        ecd.astype(np.uint8).tofile(os.path.join(os.path.join(target_root,"quantile{0}_bins{1}".format(head, bin_saved)),file_name+"_"+str(unique_time)+".npy"))
                
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