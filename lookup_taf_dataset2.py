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

def generate_event_volume(data_path, item, time_stamp_end, shape, ecd):

    feature_file = os.path.join(os.path.join(data_path,"feature"), item+ "_" + str(time_stamp_end) + ".npy")
    features = np.fromfile(feature_file, dtype=np.uint8).reshape(10, shape[0], shape[1]).astype(np.float32)
    ecd_file = os.path.join(os.path.join(data_path,ecd), item+ "_" + str(time_stamp_end) + ".npy")
    ecds = np.fromfile(ecd_file, dtype=np.uint8).reshape(10, shape[0], shape[1]).astype(np.float32)
    return features, ecds

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

def visualizeVolume(volume_, ecds,gt,dt,filename,path,time_stamp_end,tol,LABELMAP):
    img_list = []
    for i in range(0, len(volume_)):
        ecd = ecds[i]
        #ecd = volume[-1]
        #ecd = volume[-1]
        volume = volume_[i]
        img_s = 255 * np.ones((volume.shape[0], volume.shape[1], 3), dtype=np.uint8)
        #tar = volume[-1] - volume[-2]
        #tar = ecd * 2
        tar = ecd / 4 / 255
        #tar = np.where(tar > 1, (tar - 1) / 7 + 1, tar)
        #tar = tar
        #tar = np.where(tar<0,0,tar)
        #tar = np.where(tar * 10 > 1, 1, tar)
        img_0 = (60 * tar).astype(np.uint8) + 119
        #img_1 = (255 * tar).astype(np.uint8)
        #img_2 = (255 * tar).astype(np.uint8)
        img_s[:,:,0] = img_0
        #img_s[:,:,1] = img_1
        #img_s[:,:,2] = img_2
        img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
        mask = np.where(volume[:,:,None] * 8 > 1, 1, volume[:,:,None] * 8)
        #mask = np.where(volume[:,:,None] > 1, 1, volume[:,:,None])
        img_s = (mask * img_s).astype(np.uint8)
        gt = gt[gt['t']==time_stamp_end]
        draw_bboxes(img_s,gt,0,LABELMAP)
        if not (dt is None):
            dt = dt[(dt['t']>time_stamp_end-tol)&(dt['t']<time_stamp_end+tol)]
            draw_bboxes(img_s,dt,1,LABELMAP)
            path_t = os.path.join(path,filename+"_{0}_result.png".format(int(time_stamp_end)))
        else:
            path_t = os.path.join(path,filename+"_{0}_{1}.png".format(int(time_stamp_end),i))
        cv2.imwrite(path_t,img_s)
        # if not(os.path.exists(path_t)):
        #     os.mkdir(path_t)
        cv2.imwrite(path_t,img_s)
        img_list.append(img_s)
    img_all = np.stack(img_list).max(0).astype(np.uint8)
    if not (dt is None):
        dt = dt[(dt['t']>time_stamp_end-tol)&(dt['t']<time_stamp_end+tol)]
        draw_bboxes(img_all,dt,1,LABELMAP)
        path_t = os.path.join(path,filename+"_{0}_result_all.png".format(int(time_stamp_end)))
    else:
        path_t = os.path.join(path,filename+"_{0}_all.png".format(int(time_stamp_end),i))
    cv2.imwrite(path_t,img_all)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)
    parser.add_argument('-ecd', type=str)
    parser.add_argument('-exp_name', type=str, default=None)
    parser.add_argument('-tol', type = int, default=4999)
    parser.add_argument('-dataset', type = str, default="gen1")

    args = parser.parse_args()

    result_path = 'result_taf_dataset'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'train'
    item = args.item
    time_stamp_end = args.end

    if args.dataset == "gen1":
        bbox_path = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
        data_path = "/data/lbd/ATIS_Automotive_Detection_Dataset_processed/taf"
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
        bbox_path = "/datassd4t/lbd/Large_Automotive_Detection_Dataset_sampling"
        data_path = "/home/lbd/Large_Automotive_Detection_Dataset_processed/taf"
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

    data_path = os.path.join(data_path,data_folder)
    volumes, ecds = generate_event_volume(data_path, item, time_stamp_end, shape, args.ecd)
    visualizeVolume(volumes, ecds,dat_bbox,dt,item,result_path,time_stamp_end,args.tol,LABELMAP)