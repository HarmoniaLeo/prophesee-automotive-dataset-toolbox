from cgitb import small
import numpy as np
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import argparse
import cv2
from src.io import npy_events_tools

def nms(dets):
    """"Pure Python NMS baseline."""
    x1 = dets[:, 1]  #xmin
    y1 = dets[:, 2]  #ymin
    x2 = dets[:, 3]  #xmax
    y2 = dets[:, 4]  #ymax

    areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
    order = np.arange(len(dets))                        # sort bounding boxes by decreasing order

    keep = []                                             # store the final bounding boxes
    while order.size > 0:
        i = order[0]                                      #the index of the bbox with highest confidence
        keep.append(i)                                    #save it to keep
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(1e-28, xx2 - xx1)
        h = np.maximum(1e-28, yy2 - yy1)
        inter = w * h

        # Cross Area / (bbox + particular area - Cross Area)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        #reserve all the boundingbox whose ovr less than thresh
        inds = np.where(ovr <= 0)[0]
        if len(inds) > 0:
            keep.pop()
        order = order[inds + 1]

    return keep

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
        result_path2 = "/home/lbd/100-fps-event-det/log/" + args.exp_name + "/summarise_stats.npz"
        shape = [240,304]
    else:
        raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
        result_path = "/home/liubingde/100-fps-event-det/log/" + args.exp_name + "/summarise.npz"
        result_path2 = "/home/liubingde/100-fps-event-det/log/" + args.exp_name + "/summarise_stats.npz"
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

        for bbox_count,unique_time in enumerate(unique_ts):

            dt_trans = dt_bbox[(dt_bbox[:,0] >= unique_time - args.tol) & (dt_bbox[:,0] <= unique_time + args.tol)]


            flow = np.load(os.path.join("optical_flow_buffer",file_name + "_{0}.npy".format(int(unique_time))))

            dt_nms = np.zeros_like(dt_trans)
            dt_nms[:,3] = dt_trans[:,3] + dt_trans[:,1]
            dt_nms[:,4] = dt_trans[:,4] + dt_trans[:,2]

            dt_trans = dt_trans[nms(dt_nms)]

            for j in range(len(dt_trans)):
                x1, y1, x2, y2 = int(dt_trans[j,1]), int(dt_trans[j,2]), int(dt_trans[j,3] + dt_trans[j,1]), int(dt_trans[j,4] + dt_trans[j,2])

                if x1 >= shape[1]:
                    x1 = shape[1] - 1
                if x1 < 0:
                    x1 = 0
                if x2 >= shape[1]:
                    x2 = shape[1] - 1
                if x2 < 0:
                    x2 = 0
                
                if y1 >= shape[0]:
                    y1 = shape[0] - 1
                if y1 < 0:
                    y1 = 0
                if y2 >= shape[0]:
                    y2 = shape[0] - 1
                if y2 < 0:
                    y2 = 0

                density = np.sum(np.sqrt(flow[y1:y2,x1:x2,0]**2 + flow[y1:y2,x1:x2,1]**2))/((y2 - y1)*(x2 - x1) + 1e-8)
                densitys.append(density)

                dt.append(dt_trans[j])
                file_names2.append(file_name)

        #h5.close()
        pbar.update(1)
    pbar.close()
    csv_path = result_path2
    np.savez(csv_path,
        file_names = file_names2,
        dts = dt,
        densitys = densitys)