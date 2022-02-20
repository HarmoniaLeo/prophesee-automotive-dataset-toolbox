from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import h5py
import argparse
import os
import numpy as np
from src.io.psee_loader import PSEELoader
import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    args = parser.parse_args()

    data_path = "/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    output_path = "/data/ATIS_h5"

    for data_folder in ['val']:
        print(data_folder)
        final_path = os.path.join(data_path,data_folder)
        final_output_path = os.path.join(output_path,data_folder)
        files = [item for item in os.listdir(final_path) if ".dat" in item]
        h5_file = final_output_path + "_oneframe.h5"
        f = h5py.File(h5_file, 'w')
        id = 0
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        for item in files:
            item = item[:-7]
            event_file = os.path.join(final_path, item+"_td.dat")
            bbox_file = os.path.join(final_path, item+"_bbox.npy")
            
            f_bbox = open(bbox_file, "rb")
            start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
            dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
            f_bbox.close()
            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

            f_event = PSEELoader(event_file)
            
            for bbox_count,unique_time in enumerate(unique_ts):
                if ((data_folder == 'val') or (data_folder == 'test')) and (unique_time <= 500000):
                    continue
                end_time = unique_time
                end_count = f_event.seek_time(end_time)
                if end_count is None:
                    continue
                start_time = end_time - 40000
                if start_time < 0:
                    f_event.seek_event(0)
                    start_count = 0
                else:
                    start_count = f_event.seek_time(start_time)
                events_all = f_event.load_n_events(end_count - start_count)
                f.create_dataset("events/{0}".format(id), data = events_all, maxshape=(None, ), chunks=True)
                f["events/{0}".format(id)].attrs["file_name"] = item
                indices = (dat_bbox['t'] == unique_time)
                bboxes = dat_bbox[indices]
                f.create_dataset("bboxes/{0}".format(id), data = bboxes, maxshape=(None, ), chunks=True)
                f["bboxes/{0}".format(id)].attrs["file_name"] = item
                id += 1
            pbar.update(1)
        f.attrs["total"] = id
        f.close()
        pbar.close()