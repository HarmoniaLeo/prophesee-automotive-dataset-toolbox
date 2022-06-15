import numpy as np
from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import os
import cv2
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
sns.set_style("darkgrid")

def generate_event_volume(events,shape,ori_shape,C):

    x, y, c, p, features = events.T

    H, W = shape

    feature_map = np.zeros((C * H * W * 2),dtype=float)
    np.add.at(feature_map, c * H * W * 2 + y * W * 2 + x * 2 + p, features)

    volume_t = feature_map.reshape(C, H, W, 2)

    volume_t = volume_t.transpose(0,3,1,2).reshape(volume_t.shape[0] * volume_t.shape[3], volume_t.shape[1], volume_t.shape[2])

    return volume_t

LABELMAP = ["car", "pedestrian"]

def draw_bboxes(img, boxes, dt, labelmap):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes[i][1]), int(boxes[i][2]))
        size = (int(boxes[i][3]), int(boxes[i][4]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        score = boxes[i][-2]
        class_id = boxes[i][-3]
        class_name = labelmap[int(class_id)]
        color = colors[(dt+1) * 60]
        center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        cv2.rectangle(img, pt1, pt2, color, 2)
        if dt:
            cv2.rectangle(img, (pt1[0], pt1[1] - 15), (pt1[0] + 75, pt1[1]), color, -1)
            cv2.putText(img, class_name, (pt1[0]+3, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
            cv2.putText(img, "{0:.2f}".format(score), (pt1[0]+40, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        else:
            cv2.rectangle(img, (pt1[0], pt1[1] - 15), (pt1[0] + 35, pt1[1]), color, -1)
            cv2.putText(img, class_name[:3], (pt1[0]+3, pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

def visualizeVolume(volume_,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,typ):
    img_list = []
    for i in range(0, len(volume_)):
        img_s = volume_[i].astype(np.uint8)
        draw_bboxes(img_s,gt,0,LABELMAP)
        if not (dt is None):
            dt = dt[(dt['t']>time_stamp_end-tol)&(dt['t']<time_stamp_end+tol)]
            draw_bboxes(img_s,dt,1,LABELMAP)
            path_t = os.path.join(path,filename+"_{0}_{1}_result_".format(int(time_stamp_end),i)+typ+".png")
        else:
            path_t = os.path.join(path,filename+"_{0}_{1}_".format(int(time_stamp_end),i)+typ+".png")
        cv2.imwrite(path_t,img_s)
        # if not(os.path.exists(path_t)):
        #     os.mkdir(path_t)
        cv2.imwrite(path_t,img_s)
        img_list.append(img_s)
    # img_all = np.stack(img_list).max(0).astype(np.uint8)
    # if not (dt is None):
    #     dt = dt[(dt['t']>time_stamp_end-tol)&(dt['t']<time_stamp_end+tol)]
    #     draw_bboxes(img_all,dt,1,LABELMAP)
    #     path_t = os.path.join(path,filename+"_{0}_result_all.png".format(int(time_stamp_end)))
    # else:
    #     path_t = os.path.join(path,filename+"_{0}_all.png".format(int(time_stamp_end),i))
    # cv2.imwrite(path_t,img_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)
    parser.add_argument('-type', type=str, default="normal")
    parser.add_argument('-exp_name', type=str, default=None)
    parser.add_argument('-tol', type = int, default=4999)
    parser.add_argument('-dataset', type = str, default="gen1")

    args = parser.parse_args()

    result_path = 'result_lookup_dataset'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'test'
    item = args.item
    time_stamp_end = args.end

    if args.dataset == "gen1":
        bbox_path = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
        data_path = "/home/lbd/ATIS_Automotive_Detection_Dataset_processed/" + args.type
        if not (args.exp_name is None):
            result_path = "/home/lbd/100-fps-event-det/" + args.exp_name + "/summarise.npz"
        ori_shape = (240,304)
        shape = (256,320)
        LABELMAP = ["car", "pedestrian"]
    elif args.dataset == "kitti":
        bbox_path = "/home/liubingde/kitti"
        data_path = "/home/liubingde/kitti_taf"
        data_folder = 'val'
        if not (args.exp_name is None):
            result_path = "/home/lbd/100-fps-event-det/" + args.exp_name + "/summarise.npz"
        ori_shape = (375,1242)
        shape = (192,640)
        LABELMAP = ["car", "pedestrian"]
    else:
        bbox_path = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
        data_path = "/data/lbd/Large_taf"
        if not (args.exp_name is None):
            result_path = "/home/liubingde/100-fps-event-det/" + args.exp_name + "/summarise.npz"
        ori_shape = (720,1280)
        shape = (512,640)
        LABELMAP = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']

    if not (args.exp_name is None):
        bbox_file = result_path
        f_bbox = np.load(bbox_file)
        dt = f_bbox["dts"][f_bbox["file_names"]==item]
    else:
        dt = None

    final_path = os.path.join(bbox_path,data_folder)
    bbox_file = os.path.join(final_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size, _ = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    #print(target)

    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_"+str(time_stamp_end) + ".npy")
    #print(target)
    locations = np.fromfile(event_file, dtype=np.uint32)
    x = np.bitwise_and(locations, 1023).astype(int)
    y = np.right_shift(np.bitwise_and(locations, 523264), 10).astype(int)
    c = np.right_shift(np.bitwise_and(locations, 3670016), 19).astype(int)
    p = np.right_shift(np.bitwise_and(locations, 4194304), 22).astype(int)
    features = np.right_shift(np.bitwise_and(locations, 2139095040), 23).astype(int)

    events = np.stack([x, y, c, p, features], axis=1)

    if args.type == "normal":
        C = 5
    volumes = generate_event_volume(events,shape,ori_shape,C)
    print(np.quantile(volumes[volumes>0],0.05),np.quantile(volumes[volumes>0],0.2),np.quantile(volumes[volumes>0],0.5),np.quantile(volumes[volumes>0],0.75),np.quantile(volumes[volumes>0],0.95))
    visualizeVolume(volumes,dat_bbox,dt,item,result_path,time_stamp_end,args.tol,LABELMAP,args.type)