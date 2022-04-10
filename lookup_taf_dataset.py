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

def no_transform(volume):
    return volume

def minmax_transform(volume):
    volume = volume.copy()
    ecd_view = volume[...,1][volume[...,1] > -1e6]
    #q10, q90 = torch.quantile(ecd_view, torch.tensor([0.1,0.9]).to(x.device))
    q100 = np.max(ecd_view)
    q0 = np.min(ecd_view)
    volume[...,1] = np.where(volume[...,1] > -1e6, (volume[...,1] - q100) / (q100 - q0 + 1e-8) * 6, x[...,1])
    return volume

def quantile_transform(volume):
    volume = volume.copy()
    ecd_view = volume[...,1][volume[...,1] > -1e6]
    q90 = np.quantile(ecd_view, 0.90)
    q10 = np.quantile(ecd_view, 0.10)
    volume[...,1] = np.where(volume[...,1] > -1e6, volume[...,1] - q90, volume[...,1])
    volume[...,1] = np.where((volume[...,1] > -1e6) & (volume[...,1] < 0), volume[...,1]/(q90 - q10 + 1e-8) * 2, volume[...,1])
    ecd_view = volume[...,1][volume[...,1] > -1e6]
    q100 = np.max(ecd_view)
    volume[...,1] = np.where(volume[...,1] > 0, volume[...,1] / (q100 + 1e-8) * 2, volume[...,1])
    return volume

def generate_event_volume(events,shape,ori_shape):

    volumes = []
    transforms = [no_transform,quantile_transform,minmax_transform]

    x, y, t, c, z, p, features = events.T

    x, y, p, c = x.astype(int), y.astype(int), p.astype(int), c.astype(int)
    
    H, W = shape
    C = c.max() + 1

    feature_map = np.zeros((C * H * W * 2),dtype=float)
    np.add.at(feature_map, c * H * W * 2 + y * W * 2 + x * 2 + p, features)

    volume = feature_map.reshape(C, H, W, 2)
    volume[...,1] = np.where(volume[...,1] ==0, -1e6, volume[...,1] + 1)

    for transform in transforms:    
        volume_t = transform(volume)
        volume_t = torch.from_numpy(volume_t).permute(0,3,1,2).contiguous().view(volume_t.shape[0] * volume_t.shape[3], volume_t.shape[1], volume_t.shape[2])
        volume_t = torch.nn.functional.interpolate(volume_t[None,:],torch.Size(ori_shape))[0]
        volume_t = volume_t.cpu().numpy()
        volumes.append(volume_t)

    return volumes

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
        cv2.rectangle(img, pt1, pt2, color, 1)
        cv2.putText(img, class_name, (center[0], pt2[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
        cv2.putText(img, str(score), (center[0], pt1[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)

def visualizeVolume(volume,gt_i,filename,path,time_stamp_end,typ):
    ecd = volume[1:volume.shape[0]:2]
    volume = volume[:volume.shape[0]:2]
    for j in range(len(ecd)):
        img_s = 255 * np.ones((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
        ecd_view = ecd[j][ecd[j]>-1e6]
        sns.histplot(ecd_view)
        plt.xlim(-5.0,0.1)
        tar = ecd[j] + 2.0
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
        draw_bboxes(img_s,gt_i)
        path_t = os.path.join(path,filename+"_end{0}".format(int(time_stamp_end)))
        if not(os.path.exists(path_t)):
            os.mkdir(path_t)
        cv2.imwrite(os.path.join(path_t,typ+'_{0}.png'.format(j)),img_s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)

    args = parser.parse_args()

    result_path = 'result_taf_dataset'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'test'
    item = args.item
    time_stamp_end = args.end
    bbox_path = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    data_path = "/data/lbd/ATIS_taf"
    final_path = os.path.join(bbox_path,data_folder)
    bbox_file = os.path.join(final_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size, _ = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()

    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_"+str(time_stamp_end))
    #print(target)
    locations = np.fromfile(event_file + "_locations.npy", dtype=np.int32)
    x = np.bitwise_and(locations, 511).astype(np.float32)
    y = np.right_shift(np.bitwise_and(locations, 130560), 9).astype(np.float32)
    c = np.right_shift(np.bitwise_and(locations, 1966080), 17).astype(np.float32)
    p = np.right_shift(np.bitwise_and(locations, 2097152), 21).astype(np.float32)
    features = np.fromfile(event_file + "_features.npy", dtype=np.float32)

    z = np.zeros_like(c)
    t = np.zeros_like(c) + time_stamp_end
    events = np.stack([x, y, t, c, z, p, features], axis=1)

    volumes = generate_event_volume(events,(256,320),(240,304))
    gt_i = dat_bbox[dat_bbox['t']==time_stamp_end]
    for volume,typ in zip(volumes,["no_transform","quantile_transform","minmax_transform"]):
        visualizeVolume(volume,gt_i,item,result_path,time_stamp_end,typ)