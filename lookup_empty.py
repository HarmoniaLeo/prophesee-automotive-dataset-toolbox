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
import tqdm
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
    q0 = np.min(ecd_view)
    volume[...,1] = np.where(volume[...,1] > -1e8, (volume[...,1] - q100) / (q100 - q0 + 1e-8) * 6, volume[...,1])
    return volume

def quantile_transform(volume):
    volume = volume.copy()
    ecd_view = volume[...,1][volume[...,1] > -1e8]
    q90 = np.quantile(ecd_view, 0.90)
    q10 = np.quantile(ecd_view, 0.10)
    volume[...,1] = np.where(volume[...,1] > -1e8, volume[...,1] - q90, volume[...,1])
    volume[...,1] = np.where((volume[...,1] > -1e8) & (volume[...,1] < 0), volume[...,1]/(q90 - q10 + 1e-8) * 2, volume[...,1])
    ecd_view = volume[...,1][volume[...,1] > -1e8]
    q100 = np.max(ecd_view)
    volume[...,1] = np.where(volume[...,1] > 0, volume[...,1] / (q100 + 1e-8) * 2, volume[...,1])
    return volume

def generate_event_volume(events,shape,ori_shape,item):

    volumes = []
    transforms = [point1_transform,point01_transform,quantile_transform,minmax_transform]

    x, y, t, c, z, p, features = events.T

    x, y, p, c = x.astype(int), y.astype(int), p.astype(int), c.astype(int)
    
    H, W = shape
    C = c.max() + 1

    feature_map = np.zeros((C * H * W * 2),dtype=float)
    np.add.at(feature_map, c * H * W * 2 + y * W * 2 + x * 2 + p, features)

    volume = feature_map.reshape(C, H, W, 2)
    volume[...,1] = np.where(volume[...,1] ==0, -1e8, volume[...,1] + 1)

    if np.sum(volume[...,1] > -1e8)==0:
        print(item)

LABELMAP = ["car", "pedestrian"]

def draw_bboxes(img, boxes, dt = 0, labelmap=LABELMAP):
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

def visualizeVolume(volume,gt_i,filename,path,time_stamp_end,typ):
    ecd = volume[1:volume.shape[0]:2]
    volume = volume[:volume.shape[0]:2]
    img_s = 255 * np.ones((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
    tar = ecd[-1] + 2.0
    #tar = volume[-1] - volume[-2]
    tar = tar / 2.0
    tar = np.where(tar<0,0,tar)
    #tar = np.where(tar * 10 > 1, 1, tar)
    img_0 = (60 * tar).astype(np.uint8) + 119
    #img_1 = (255 * tar).astype(np.uint8)
    #img_2 = (255 * tar).astype(np.uint8)
    img_s[:,:,0] = img_0
    #img_s[:,:,1] = img_1
    #img_s[:,:,2] = img_2
    img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
    #draw_bboxes(img_s,gt_i)
    path_t = os.path.join(path,filename+"_end{0}".format(int(time_stamp_end)))
    if not(os.path.exists(path_t)):
        os.mkdir(path_t)
    cv2.imwrite(os.path.join(path_t,typ+'.png'),img_s)

if __name__ == '__main__':
    data_path = "/data/lbd/ATIS_taf_all"
    for data_folder in ["train","val","test"]:
        file_dir = os.path.join(data_path, data_folder)
        files = os.listdir(file_dir)
        # Remove duplicates (.npy and .dat)
        files = [time_seq_name[:-14] for time_seq_name in files
                      if (time_seq_name[-3:] == 'npy') and ("locations" in time_seq_name)]
        pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
        for item in files:
            event_file = os.path.join(file_dir, item)
            #print(target)
            locations = np.fromfile(event_file + "_locations.npy", dtype=np.int32)
            x = np.bitwise_and(locations, 1023).astype(np.float32)
            y = np.right_shift(np.bitwise_and(locations, 523264), 10).astype(np.float32)
            c = np.right_shift(np.bitwise_and(locations, 7864320), 19).astype(np.float32)
            p = np.right_shift(np.bitwise_and(locations, 8388608), 23).astype(np.float32)
            features = np.fromfile(event_file + "_features.npy", dtype=np.float32)

            z = np.zeros_like(c)
            t = np.zeros_like(c)
            events = np.stack([x, y, t, c, z, p, features], axis=1)
            volumes = generate_event_volume(events,(256,320),(240,304),item)
            pbar.update(1)
        pbar.close()