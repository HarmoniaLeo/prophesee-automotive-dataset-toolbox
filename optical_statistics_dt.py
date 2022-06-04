from cgitb import small
import numpy as np
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import argparse
import cv2
from src.io import npy_events_tools

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-tol', type = int, default=4999)

    args = parser.parse_args()
    mode = "test"

    if args.dataset == "gen1":
        raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
        result_path = "/home/lbd/100-fps-event-det/log/" + args.exp_name + "/summarise.npz"
        shape = [240,304]
    else:
        raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
        result_path = "/home/liubingde/100-fps-event-det/log/" + args.exp_name + "/summarise.npz"
        shape = [720,1280]
    
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

    densitys = []
    file_names2 = []
    dt = []

    f_bbox = np.load(result_path)
    dts = f_bbox["dts"]
    file_names = f_bbox["file_names"]

    for i_file, file_name in enumerate(files):        

        bbox_file = os.path.join(root, file_name + '_bbox.npy')
        # if os.path.exists(volume_save_path):
        #     continue
        #h5 = h5py.File(volume_save_path, "w")
        f_bbox = open(bbox_file, "rb")
        start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
        gt_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        gt_bbox = rfn.structured_to_unstructured(gt_bbox)

        unique_ts, unique_indices = np.unique(gt_bbox[:,0], return_index=True)

        dt_bbox = dts[file_names == file_name]

        # np.clip(dt_bbox[:, 1], 0, shape[1], out=dt_bbox[:, 1])
        # np.clip(dt_bbox[:, 2], 0, shape[0], out=dt_bbox[:, 2])
        # np.clip(dt_bbox[:, 3] + dt_bbox[:, 1], 0, shape[1], out=dt_bbox[:, 3])
        # np.clip(dt_bbox[:, 4] + dt_bbox[:, 2], 0, shape[0], out=dt_bbox[:, 4])
        # dt_bbox = dt_bbox[(dt_bbox[:, 3] - dt_bbox[:, 1] > 0)&(dt_bbox[:, 4] - dt_bbox[:, 2] > 0)]

        for bbox_count,unique_time in enumerate(unique_ts):

            dt_trans = dt_bbox[(dt_bbox[:,0] >= unique_time - args.tol) & (dt_bbox[:,0] <= unique_time + args.tol)]

            flow = np.load(os.path.join("optical_flow_buffer",file_name + "_{0}.npy".format(int(unique_time))))

            for j in range(len(dt_trans)):
                x, y, w, h = int(dt_trans[j,1]), int(dt_trans[j,2]), int(dt_trans[j,3]), int(dt_trans[j,4])

                density = np.sum(np.sqrt(flow[y:y+h,x:x+w,0]**2 + flow[y:y+h,x:x+w,1]**2))/(w * h + 1e-8)
                densitys.append(density)

                dt.append(dt_trans[j])
                file_names2.append(file_name)

        #h5.close()
        pbar.update(1)
    pbar.close()
    csv_path = result_path
    np.savez(csv_path,
        file_names = file_names2,
        dts = dt,
        densitys = densitys)