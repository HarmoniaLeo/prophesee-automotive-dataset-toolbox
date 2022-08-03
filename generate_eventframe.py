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


def generate_eventframe((events, shape):
    """
    Events: N x 4, where cols are x, y, t, polarity, and polarity is in {0,1}. x and y correspond to image
    coordinates u and v.
    """
    H, W = shape
    x, y, t, p = events.unbind(-1)

    x, y, p = x.long(), y.long(), p.long()

    img = torch.zeros((H * W * 2,)).float().to(x.device)

    img.index_add_(0, 2 * x + 2 * W * y + p, torch.zeros_like(x).float()+0.05)

    img = torch.where(img > 1, torch.ones_like(img).float(), img)

    histogram = img.view((H, W, 2)).permute(2, 0, 1).contiguous()

    return histogram * 255

def generate_frame(events, shape, events_window = 50000, volume_bins=5):
    H, W = shape

    x, y, t, p = events.unbind(-1)

    x, y, p = x.long(), y.long(), p.long()

    t_star = (volume_bins * t.float())[:,None,None]
    channels = volume_bins

    adder = torch.stack([torch.arange(channels),torch.arange(channels)],dim = 1).to(x.device)[None,:,:] + 1   #1, 2, 2
    adder = (1 - torch.abs(adder-t_star)) * torch.stack([p,1 - p],dim=1)[:,None,:]  #n, 2, 2
    adder = torch.where(adder>=0,adder,torch.zeros_like(adder)).view(adder.shape[0], channels * 2) #n, 4

    img = torch.zeros((H * W, volume_bins * 2)).float().to(x.device)
    img.index_add_(0, x + W * y, adder)
    img = img.view(H * W, volume_bins, 2)

    img_viewed = img.view((H, W, img.shape[1] * 2)).permute(2, 0, 1).contiguous()

    # print(torch.quantile(img_viewed[img_viewed>0],0.95))

    img_viewed = img_viewed / 5 * 255

    return img_viewed

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
    #events_window = 500000
    events_windows = [200000, 100000, 50000]

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    for mode in ["train","val","test"]:
        file_dir = os.path.join(raw_dir, mode)
        root = file_dir
        label_root = os.path.join(label_dir, mode)
        target_root = os.path.join(target_dir, mode)
        if not os.path.exists(target_root):
            os.makedirs(target_root)
        try:
            files = os.listdir(file_dir)
        except Exception:
            continue
        # Remove duplicates (.npy and .dat)
        # files = files[int(2*len(files)/3):]
        #files = files[int(len(files)/3):]g
        files = [time_seq_name[:-7] for time_seq_name in files
                        if time_seq_name[-3:] == 'dat']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

        for i_file, file_name in enumerate(files):

            if not file_name == "17-04-13_15-05-43_3599500000_3659500000":
                continue
            # if not file_name == "moorea_2019-06-26_test_02_000_976500000_1036500000":
            #     continue

            model = load_model(args.path_to_model)
            device = torch.device('cuda:0')
            model = model.to(device)
            model.eval()
            reconstructor = ImageReconstructor(model, shape[0], shape[1], model.num_bins, args)
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
            count_upper_bound = -100000000
            memory = None

            for bbox_count,unique_time in enumerate(unique_ts):
                # if os.path.exists(os.path.join(os.path.join(os.path.join(target_dir,"e2vid"), mode),file_name+"_"+str(unique_time)+".npy")):
                #     continue
                end_time = int(unique_time)
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_count = int(end_count - np.max(events_windows))
                if start_count < 0:
                    start_count = 0
                if start_count <= count_upper_bound:
                    start_count = count_upper_bound
                
                dat_event = f_event
                dat_event.seek_event(start_count)

                events = dat_event.load_n_events(int(end_count - start_count))
                del dat_event
                events = rfn.structured_to_unstructured(events)[:, [0, 1, 2, 3]].astype(float)

                if not memory is None:
                    events = np.concatenate([memory, events])

                memory = events[-np.max(events_windows):]
                count_upper_bound = end_count

                for events_window in events_windows:
                
                    events_ = events[-events_window:].clone()
                    
                    if target_shape[0] < shape[0]:
                        events_[:,0] = events_[:,0] * rw
                        events_[:,1] = events_[:,1] * rh
                        volume = generate_eventframe(events_, target_shape)
                    else:
                        volume = generate_eventframe(events_, shape)
                        volume = torch.nn.functional.interpolate(volume[None,:,:,:], size = target_shape, mode='nearest')[0]

                    save_dir = os.path.join(target_dir,"frame{0}".format(events_window))
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_dir = os.path.join(save_dir, mode)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    
                    ecd = volume.cpu().numpy().copy()[0]
                    
                    ecd.astype(np.uint8).tofile(os.path.join(save_dir,file_name+"_"+str(unique_time)+".npy"))
                            
                torch.cuda.empty_cache()
            #h5.close()
            pbar.update(1)
        pbar.close()
        # if mode == "test":
        #     np.save(os.path.join(root, 'total_volume_time.npy'),np.array(total_volume_time))
        #     np.save(os.path.join(root, 'total_taf_time.npy'),np.array(total_taf_time))
        #h5.close()