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


def taf_cuda(x, y, t, p, shape, lamdas):
    H, W = shape

    t_img = torch.zeros((2, H, W)).float().to(x.device) - 2000000
    t_img.index_put_(indices= [p, y, x], values= t)

    t_imgs = []
    for lamda in lamdas:
        t_img_ = torch.exp(lamda * t_img)
        t_imgs.append(t_img_)
    ecd = torch.stack(t_imgs, 0)

    ecd_viewed = ecd.view(len(lamdas) * 2, H, W) * 255

    #print(generate_volume_time, filter_time, generate_encode_time)
    return ecd_viewed

def generate_leaky_cuda(events, shape, lamdas):
    x, y, t, p = events.unbind(-1)

    x, y, t, p = x.long(), y.long(), t.float(), p.long()
    
    histogram_ecd = taf_cuda(x, y, t, p, shape, lamdas)

    return histogram_ecd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description='visualize one or several event files along with their boxes')
    parser.add_argument('-raw_dir', type=str)
    parser.add_argument('-label_dir', type=str)
    parser.add_argument('-target_dir', type=str)
    parser.add_argument('-dataset', type=str, default="gen4")

    args = parser.parse_args()
    raw_dir = args.raw_dir
    label_dir = args.label_dir
    target_dir = args.target_dir
    dataset = args.dataset

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
    events_window = 5000000

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)


    #lamdas = [0.00001, 0.000005, 0.0000025, 0.000001]
    lamdas = [0.000001]

    for mode in ["train","val","test"]:
        file_dir = os.path.join(raw_dir, mode)
        root = file_dir
        label_dir = os.path.join(label_dir, mode)
        label_root = label_dir
        target_root = os.path.join(target_dir, mode)
        if not os.path.exists(target_root):
            os.makedirs(target_root)
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
            # if not file_name == "17-04-13_15-05-43_3599500000_3659500000":
            #     continue
            # if not file_name == "moorea_2019-06-26_test_02_000_976500000_1036500000":
            #     continue
            event_file = os.path.join(root, file_name + '_td.dat')
            bbox_file = os.path.join(label_root, file_name + '_bbox.npy')
            #h5 = h5py.File(volume_save_path, "w")
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = psee_loader.PSEELoader(event_file)

            #min_event_count = f_event.event_count()
            time_upper_bound = -100000000
            count_upper_bound = 0
            memory = None

            for bbox_count,unique_time in enumerate(unique_ts):
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_time = int(f_event.current_time)
                start_time = end_time - events_window
                if start_time < 0:
                    start_count = f_event.seek_time(0)
                else:
                    start_count = f_event.seek_time(start_time)
                
                if (start_count is None) or (start_time < 0):
                    start_count = 0
                
                if start_time <= time_upper_bound:
                    start_count = count_upper_bound
                
                dat_event = f_event
                dat_event.seek_event(start_count)

                events = dat_event.load_n_events(int(end_count - start_count))
                del dat_event
                events = torch.from_numpy(rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)).cuda()

                if not memory is None:
                    events = torch.cat([memory, events])
                
                events = events[events[:, 2] > unique_time - events_window]
                memory = events.clone()

                time_upper_bound = unique_time
                count_upper_bound = end_count

                events[:, 2] = events[:, 2] - unique_time
                volume = generate_leaky_cuda(events, shape, lamdas)

                volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]
                volume = volume.view(len(lamdas), 2, target_shape[0], target_shape[1])
                for j,i in enumerate(lamdas):
                    save_dir = os.path.join(target_dir,"leaky{0}".format(i))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_dir = os.path.join(save_dir, mode)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    ecd = volume[j].cpu().numpy().copy()
                    
                    ecd.astype(np.uint8).tofile(os.path.join(save_dir,file_name+"_"+str(unique_time)+".npy"))
                            
                torch.cuda.empty_cache()
            #h5.close()
            pbar.update(1)
        pbar.close()
        # if mode == "test":
        #     np.save(os.path.join(root, 'total_volume_time.npy'),np.array(total_volume_time))
        #     np.save(os.path.join(root, 'total_taf_time.npy'),np.array(total_taf_time))
        #h5.close()