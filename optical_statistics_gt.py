from cgitb import small
import numpy as np
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-dataset', type=str)

    args = parser.parse_args()
    mode = "test"

    if args.dataset == "gen1":
        raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
        shape = [240,304]
    else:
        raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
        shape = [720,1280]
    
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
        bbox_file = os.path.join(root, file_name + '_bbox.npy')
        # if os.path.exists(volume_save_path):
        #     continue
        #h5 = h5py.File(volume_save_path, "w")
        f_bbox = open(bbox_file, "rb")
        start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        dat_bbox = rfn.structured_to_unstructured(dat_bbox)

        unique_ts, unique_indices = np.unique(dat_bbox[:,0], return_index=True)

        np.clip(dat_bbox[:, 1], 0, shape[1], out=dat_bbox[:, 1])
        np.clip(dat_bbox[:, 2], 0, shape[0], out=dat_bbox[:, 2])
        np.clip(dat_bbox[:, 3], 0, shape[1], out=dat_bbox[:, 3])
        np.clip(dat_bbox[:, 4], 0, shape[0], out=dat_bbox[:, 4])
        dat_bbox = dat_bbox[(dat_bbox[:, 2] - dat_bbox[:, 0] > 0)&(dat_bbox[:, 3] - dat_bbox[:, 1] > 0)]

        for bbox_count,unique_time in enumerate(unique_ts):

            gt_trans = dat_bbox[dat_bbox[:,0] == unique_time]

            flow = np.load(os.path.join("optical_flow_buffer",file_name + "_{0}.npy".format(int(unique_time))))

            for j in range(len(gt_trans)):
                x, y, w, h = gt_trans[j,1], gt_trans[j,2], gt_trans[j,3], gt_trans[j,4]
                file_names.append(file_name)
                gt.append(gt_trans[j])

                density = np.sum(np.sqrt(flow[int(x):int(x+w),int(y):int(y+h),0]**2 + flow[int(x):int(x+w),int(y):int(y+h),1]**2))/(w*h + 1e-8)
                densitys.append(density)

        #h5.close()
        pbar.update(1)
    pbar.close()
    csv_path = os.path.join(result_path,"gt_"+args.dataset+".npz")
    print([np.quantile(densitys,q/100) for q in range(0,100,5)])
    np.savez(csv_path,
        file_names = file_names,
        gts = gt,
        densitys = densitys)