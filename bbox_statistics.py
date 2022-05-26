from cgitb import small
import numpy as np
from src.io import npy_events_tools
import tqdm
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-dataset', type=str)

    args = parser.parse_args()

    if args.dataset == "gen1":
        raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    else:
        raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
    
    for mode in ["train","val","test"]:
        file_dir = os.path.join(raw_dir, mode)
        root = file_dir
        #h5 = h5py.File(raw_dir + '/ATIS_taf_'+mode+'.h5', 'w')
        files = os.listdir(file_dir)

        # Remove duplicates (.npy and .dat)
        files = [time_seq_name[:-7] for time_seq_name in files
                        if time_seq_name[-3:] == 'dat']

        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

        class_counts = [0, 0, 0, 0, 0, 0, 0]
        small_counts = 0
        medium_counts = 0
        large_counts = 0
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

            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            for bbox_count,unique_time in enumerate(unique_ts):

                gt_trans = dat_bbox[dat_bbox['t'] == unique_time]

                for j in range(len(gt_trans)):
                    x, y, w, h = gt_trans['x'][j], gt_trans['y'][j], gt_trans['w'][j], gt_trans['h'][j]
                    area = w * h
                    if area < 32*32:
                        small_counts+=1
                    elif area < 96*96:
                        medium_counts+=1
                    else:
                        large_counts+=1
                    class_counts[gt_trans[j][5]] += 1

            #h5.close()
            pbar.update(1)
        print(class_counts, small_counts, medium_counts, large_counts)
        pbar.close()