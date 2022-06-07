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

def point1_transform(volume):
    volume = volume.copy()
    volume[...,1] = np.where(volume[...,1] > -1e8, 0.1 * volume[...,1], volume[...,1])
    return volume

def point01_transform(volume):
    volume = volume.copy()
    volume[...,1] = np.where(volume[...,1] > -1e8, 0.01 * volume[...,1], volume[...,1])
    return volume

def minmax_transform(volume):
    volume = volume.copy()
    ecd_view = volume[...,1][volume[...,1] > -1e8]
    #q10, q90 = torch.quantile(ecd_view, torch.tensor([0.1,0.9]).to(x.device))
    q100 = np.max(ecd_view)
    #q0 = np.min(ecd_view)
    q10 = np.quantile(ecd_view, 0.10)
    volume[...,1] = np.where((volume[...,1] > -1e8), (volume[...,1] - q100) / (q100 - q10 + 1e-8) * 6, volume[...,1])
    return volume

def quantile_transform(volume):
    volume = volume.copy()
    ecd_view = volume[...,1][volume[...,1] > -1e8]
    q90 = np.quantile(ecd_view, 0.9)
    q10 = np.quantile(ecd_view, 0.10)
    volume[...,1] = np.where(volume[...,1] > -1e8, volume[...,1] - q90, volume[...,1])
    volume[...,1] = np.where((volume[...,1] > -1e8) & (volume[...,1] < 0), volume[...,1]/(q90 - q10 + 1e-8) * 6, volume[...,1])
    ecd_view = volume[...,1][volume[...,1] > -1e8]
    q100 = np.max(ecd_view)
    volume[...,1] = np.where(volume[...,1] > 0, volume[...,1] / (q100 + 1e-8) * 2, volume[...,1])
    return volume

def generate_event_volume(events,shape,ori_shape):

    volumes = []
    #transforms = [point1_transform,point01_transform,quantile_transform,minmax_transform]
    #transforms = [minmax_transform]
    transforms = [quantile_transform]

    x, y, t, c, z, p, features = events.T

    x, y, p, c = x.astype(int), y.astype(int), p.astype(int), c.astype(int)
    
    H, W = shape
    C = c.max() + 1

    feature_map = np.zeros((C * H * W * 2),dtype=float)
    np.add.at(feature_map, c * H * W * 2 + y * W * 2 + x * 2 + p, features)

    volume = feature_map.reshape(C, H, W, 2)
    volume[...,1] = np.where(volume[...,1] ==0, -1e8, volume[...,1] + 1)

    for transform in transforms:    
        volume_t = transform(volume)
        volume_t = torch.from_numpy(volume_t).permute(0,3,1,2).contiguous().view(volume_t.shape[0] * volume_t.shape[3], volume_t.shape[1], volume_t.shape[2])
        volume_t = torch.nn.functional.interpolate(volume_t[None,:],torch.Size(ori_shape))[0]
        volume_t = volume_t.cpu().numpy()
        volumes.append(volume_t)

    return volumes

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

def visualizeVolume(volume,gt,dt,filename,path,time_stamp_end,tol,LABELMAP):
    ecd = np.exp(volume[-1])
    #ecd = volume[-1]
    #ecd = volume[-1]
    volume = volume[-2]
    img_s = 255 * np.ones((volume.shape[0], volume.shape[1], 3), dtype=np.uint8)
    #tar = volume[-1] - volume[-2]
    #tar = ecd * 2
    tar = ecd / 4
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
        path_t = os.path.join(path,filename+"_{0}.png".format(int(time_stamp_end)))
    cv2.imwrite(path_t,img_s)
    # if not(os.path.exists(path_t)):
    #     os.mkdir(path_t)
    cv2.imwrite(path_t,img_s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)
    parser.add_argument('-window', type=int, default=50000)
    parser.add_argument('-exp_name', type=str, default=None)
    parser.add_argument('-tol', type = int, default=4999)
    parser.add_argument('-dataset', type = str, default="gen1")

    args = parser.parse_args()

    result_path = 'result_taf_dataset'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'test'
    item = args.item
    time_stamp_start = args.end - args.window
    time_stamp_end = args.end

    if args.dataset == "gen1":
        bbox_path = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
        data_path = "/data/lbd/ATIS_taf_all"
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
        data_path = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
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

    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_td.dat")
    bbox_file = os.path.join(final_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size, _ = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    #print(target)

    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_"+str(time_stamp_end))
    #print(target)
    locations = np.fromfile(event_file + "_locations.npy", dtype=np.int32)
    x = np.bitwise_and(locations, 1023).astype(np.float32)
    y = np.right_shift(np.bitwise_and(locations, 523264), 10).astype(np.float32)
    c = np.right_shift(np.bitwise_and(locations, 7864320), 19).astype(np.float32)
    p = np.right_shift(np.bitwise_and(locations, 8388608), 23).astype(np.float32)
    features = np.fromfile(event_file + "_features.npy", dtype=np.float32)

    z = np.zeros_like(c)
    t = np.zeros_like(c) + time_stamp_end
    events = np.stack([x, y, t, c, z, p, features], axis=1)

    volumes = generate_event_volume(events,shape,ori_shape)
    visualizeVolume(volumes[0],dat_bbox,dt,item,result_path,time_stamp_end)