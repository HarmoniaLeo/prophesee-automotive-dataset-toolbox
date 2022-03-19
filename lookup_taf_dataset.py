import numpy as np
from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import os
import cv2
import argparse
from poisson import poissoned_events
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style("darkgrid")

def generate_event_volume(events,shape,ori_shape):
    rh = ori_shape[0]/shape[0]
    rw = ori_shape[1]/shape[1]

    x, y, t, c, z, p, features = events.T
    x = x * rw
    y = y * rh

    x, y, p, c = x.astype(int), y.astype(int), p.astype(int), c.astype(int)
    
    H, W = shape
    C = c.max() + 1

    feature_map = np.zeros((C * H * W * 2),dtype=float)
    np.add.at(feature_map, c * H * W * 2 + y * W * 2 + x * 2 + p, features)

    volume = feature_map.reshape(C, H, W, 2)
    volume[...,1] = np.where(volume[...,1] ==0, -1e6, volume[...,1] + 1)

    return volume[...,0], volume[...,1]

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

def visualizeVolume(volume,ecd,gt_i,filename,path,time_stamp_end):
    for j in range(len(ecd)):
        img_s = 255 * np.ones((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
        ecd_view = ecd[j][ecd[j]>-1e6]
        print(j)
        for i in range(0,100,10):
            print(np.percentile(ecd_view,))
        tar = ecd[j] - ecd[j][ecd[j]>-1e6].min(axis=0).min(axis=0)
        tar = tar / tar.max(axis=0).max(axis=0)
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
        cv2.imwrite(os.path.join(path_t,'{0}.png'.format(j)),img_s)

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
    start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()

    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_"+str(time_stamp_end)+".npz")
    #print(target)
    buffer = np.load(event_file,allow_pickle=True)
    locations = buffer["locations"]
    features = buffer["features"]
    c, y, x, p = locations
    z = np.zeros_like(c)
    t = np.zeros_like(c) + time_stamp_end
    events = np.stack([x, y, t, c, z, p, features], axis=1)

    volume, ecd = generate_event_volume(events,(256,320),(240,304))
    gt_i = dat_bbox[dat_bbox['t']==time_stamp_end]
    visualizeVolume(volume,ecd,gt_i,item,result_path,time_stamp_end)