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

LABELMAP = ["car", "pedestrian"]

def generate_event_volume(events,shape,ori_shape):

    x, y, t, c, z, p, features = events.T

    x, y, p, c = x.astype(int), y.astype(int), p.astype(int), c.astype(int)
    
    H, W = shape
    C = c.max() + 1

    feature_map = np.zeros((C * H * W * 2),dtype=float)
    np.add.at(feature_map, c * H * W * 2 + y * W * 2 + x * 2 + p, features)

    volume_t = feature_map.reshape(C, H, W, 2)
   
    volume_t = torch.from_numpy(volume_t).permute(0,3,1,2).contiguous().view(volume_t.shape[0] * volume_t.shape[3], volume_t.shape[1], volume_t.shape[2])
    volume_t = torch.nn.functional.interpolate(volume_t[None,:],torch.Size(ori_shape))[0]
    volume_t = volume_t.cpu().numpy()

    return volume_t[0]

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

def visualizeVolume(volume,gt,dt,filename,path,time_stamp_end,tol,LABELMAP,lamda):
    img_s = 255 * np.ones((volume.shape[0], volume.shape[1], 3), dtype=np.uint8)
    quant = np.quantile(volume, 0.95)
    quant = 2
    volume = np.where(volume>quant,quant,volume)
    tar = volume / volume.max()
    #tar = np.where(tar * 10 > 1, 1, tar)
    img_0 = (60 * tar).astype(np.uint8) + 119
    #img_1 = (255 * tar).astype(np.uint8)
    #img_2 = (255 * tar).astype(np.uint8)
    img_s[:,:,0] = img_0
    #img_s[:,:,1] = img_1
    #img_s[:,:,2] = img_2
    img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
    draw_bboxes(img_s,gt,0,LABELMAP)
    print(lamda)
    if not (dt is None):
        dt = dt[(dt['t']>time_stamp_end-tol)&(dt['t']<time_stamp_end+tol)]
        draw_bboxes(img_s,dt,1,LABELMAP)
        path_t = os.path.join(path,filename+"_end{0}_lamda{1}_result.png".format(int(time_stamp_end),lamda))
    else:
        path_t = os.path.join(path,filename+"_end{0}_lamda{1}.png".format(int(time_stamp_end),lamda))
    cv2.imwrite(path_t,img_s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)
    parser.add_argument('-window', type=int, default=50000)
    parser.add_argument('-exp_name', type=str, default=None)
    parser.add_argument('-tol', type = int, default=4999)
    parser.add_argument('-short', type = str, default="True")
    parser.add_argument('-dataset', type = str, default="gen1")

    args = parser.parse_args()

    result_path = 'result_leaky_dataset'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'train'
    item = args.item
    time_stamp_start = args.end - args.window
    time_stamp_end = args.end

    if args.dataset == "gen1":
        bbox_path = "/data2/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
        if args.short == "True":
            data_path_short = "/data2/lbd/ATIS_leaky"
        else:
            data_path_long = "/data2/lbd/ATIS_leaky_long"
        if not (args.exp_name is None):
            result_path = "/home/lbd/100-fps-event-det/" + args.exp_name + "/summarise.npz"
        ori_shape = (240,304)
        shape = (256,320)
        LABELMAP = ["car", "pedestrian"]
    else:
        bbox_path = "/data2/lbd/Large_Automotive_Detection_Dataset_sampling"
        if args.short == "True":
            data_path_short = "/data2/lbd/Large_leaky"
        else:
            data_path_long = "/data2/lbd/Large_leaky_long"
        if not (args.exp_name is None):
            result_path = "/home/lbd/100-fps-event-det/" + args.exp_name + "/summarise.npz"
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
    event_file = os.path.join(final_path, item+"_td.dat")
    bbox_file = os.path.join(final_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size, _ = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    #print(target)

    final_path = os.path.join(data_path_short, data_folder)

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

    volume = generate_event_volume(events,shape,ori_shape)
    if args.short == "True":
        lamda = 0.0001
    else:
        lamda = 0.000001
    visualizeVolume(volume,dat_bbox,dt,item,result_path,time_stamp_end,args.tol,LABELMAP,lamda)