from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import h5py
import argparse
import os
import numpy as np
from src.io.psee_loader import PSEELoader

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
    h5_file = os.path.join(final_path, item+"_h5.h5")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

    f_event = PSEELoader(event_file)

    f = h5py.File(h5_file, 'w')

    time_upperbound = -100000
    count_upperbound = 0
    id = -1
    for bbox_count,unique_time in enumerate(unique_ts):
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
        if start_time < 0:
            start_count = 0
        if start_time > time_upperbound:
            id += 1
            events_all = f_event.load_n_events(end_count - start_count)
            x,y,t,p = events_all['x'], events_all['y'], events_all['t'], events_all['p']
            events_all = np.stack([x.astype(int), y.astype(int), t, p], axis=-1)
            f.create_dataset("events/{0}".format(id), data = events_all, maxshape=(None, events_all.shape[1]), chunks=True)
            indices = (unique_ts == unique_time)
            bboxes = dat_bbox[indices]
            f.create_dataset("bboxes/{0}".format(id), data = bboxes, maxshape=(None, bboxes.shape[1]), chunks=True)
        else:
            events_all = f_event.load_n_events(end_count - count_upperbound)
            x,y,t,p = events_all['x'], events_all['y'], events_all['t'], events_all['p']
            events_all = np.stack([x.astype(int), y.astype(int), t, p], axis=-1)
            f["events/{0}".format(id)].resize((f["events/{0}".format(id)].shape[0] + len(events_all),events_all.shape[1]))
            f["events/{0}".format(id)][-len(events_all):] = events_all
            indices = (unique_ts == unique_time)
            bboxes = dat_bbox[indices]
            f["bboxes/{0}".format(id)].resize((f["bboxes/{0}".format(id)].shape[0] + len(bboxes),bboxes.shape[1]))
            f["bboxes/{0}".format(id)][-len(bboxes):] = bboxes
        time_upperbound = end_time
        count_upperbound = end_count
    f.close()