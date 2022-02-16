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
    h5_file = os.path.join(final_path, item+"_h5.h5")

    f = h5py.File(h5_file, 'w')

    total_time = 0
    for idx,data in enumerate(f.keys()):
        tick = time.time()
        if (unique_time <= 500000):
            continue
        end_time = unique_time
        end_count = f_event.seek_time(end_time)
        start_count = end_count - 200000
        if start_count < 0:
            start_count = 0
        f_event.seek_event(start_count)
        start_time = f_event.current_time
        bins = (end_time - start_time)//50000 + 1
        start_time = end_time - bins * 50000
        events_all = f_event.load_n_events(end_count - start_count)
        total_time += time.time() - tick
    print(total_time/len(unique_ts))