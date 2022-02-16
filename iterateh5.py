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

    f = h5py.File(h5_file, 'r')

    total_time = 0
    for idx in range(f.attrs["total"]):
        tick = time.time()
        f1 = h5py.File(h5_file, 'r')
        events = f1["events/{0}".format(idx)]
        bboxes = f1["bboxes/{0}".format(idx)]
        total_time += time.time() - tick
    print(total_time/f.attrs["total"])