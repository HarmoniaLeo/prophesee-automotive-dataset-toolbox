from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import h5py
import argparse
import os
import numpy as np
from src.io.psee_loader import PSEELoader
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    args = parser.parse_args()

    data_folder = 'test'
    item = args.item
    data_path = "/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_td.dat")
    bbox_file = os.path.join(final_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

    total_time = 0
    for bbox_count,unique_time in enumerate(unique_ts):
        tick = time.time()
        if (unique_time <= 500000):
            continue
        end_time = unique_time
        f_bbox = open(bbox_file, "rb")
        start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()
        f_event = PSEELoader(event_file)
        end_count = f_event.seek_time(end_time)
        events_all = f_event.load_delta_t(50000)
        total_time += time.time() - tick
    print(total_time/len(unique_ts))