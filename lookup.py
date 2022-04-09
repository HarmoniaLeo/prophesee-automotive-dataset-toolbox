import numpy as np
from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import os
import cv2
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style("darkgrid")

def generate_event_volume(events,shape,bins=5):
    H, W = shape
    x, y, t, p = events.T

    x, y, p = x.astype(np.int), y.astype(np.int), p.astype(np.int)

    try:
        t_min = np.min(t)
        t_max = np.max(t)
        t = t.astype(np.float)
        t = (t-t_min)/(t_max-t_min+1e-8)

        t_star = bins * t[:,None]
        
        xpos = x[p == 1]
        ypos = y[p == 1]
        adderpos = np.arange(1,bins+1)[None,:]
        adderpos = 1 - np.abs(adderpos-t_star[p == 1])
        adderpos = np.where(adderpos>=0,adderpos,0)

        xneg = x[p == 0]
        yneg = y[p == 0]
        adderneg = np.arange(1,bins+1)[None,:]
        adderneg = 1 - np.abs(adderneg-t_star[p == 0])
        adderneg = np.where(adderneg>=0,adderneg,0)

        img_pos = np.zeros((H * W , bins),dtype=float)
        np.add.at(img_pos, W * ypos + xpos, adderpos)
        img_neg = np.zeros((H * W , bins),dtype=float)
        np.add.at(img_neg, W * yneg + xneg, adderneg)
    except Exception:
        img_pos = np.zeros((H * W , bins),dtype=float)
        img_neg = np.zeros((H * W , bins),dtype=float)

    histogram = np.concatenate([img_neg, img_pos], -1).reshape((H, W, bins*2)).transpose(2,0,1)

    return histogram

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

def visualizeVolume(volume,gt,filename,path,time_stamp_start,time_stamp_end):
    img = 127 * np.ones((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
    gt = gt[(gt['t']>=time_stamp_start)&(gt['t']<=time_stamp_end)]
    for i in range(0,volume.shape[0],2):
        c_p = volume[i+1]
        c_p = 127 * c_p / (np.percentile(c_p,0.99)+1e-8)
        c_p = np.where(c_p>127, 127, c_p)
        c_n = volume[i]
        c_n = 127 * c_n / (np.percentile(c_p,0.99)+1e-8)
        c_n = np.where(c_n>127, 127, c_n)
        img_s = img + c_p[:,:,None].astype(np.uint8) - c_n[:,:,None].astype(np.uint8)
        draw_bboxes(img_s,gt,0,LABELMAP)
        path_t = os.path.join(path,filename+"_{0}".format(int(time_stamp_end)))
        if not(os.path.exists(path_t)):
            os.mkdir(path_t)
        cv2.imwrite(os.path.join(path_t,'{0}.png'.format(i)),img_s)
    density = (np.sum(np.sum(np.sum(volume,axis=0)>0,axis=0),axis=0)/(volume.shape[1]*volume.shape[2]))
    density_p = (np.sum(np.sum(np.sum(volume[1::2]>0,axis=0),axis=0))/(volume.shape[1]*volume.shape[2]))
    density_n = (np.sum(np.sum(np.sum(volume[0::2]>0,axis=0),axis=0))/(volume.shape[1]*volume.shape[2]))
    total_area = 0
    total_points = 0
    gt_trans = gt
    max_density = 0
    for j in range(len(gt_trans)):
        x, y, w, h = gt_trans['x'][j], gt_trans['y'][j], gt_trans['w'][j], gt_trans['h'][j]
        area = w * h
        points = np.sum(np.sum(np.sum(volume[int(y):int(y+h),int(x):int(x+w)],axis=0)>0,axis=0),axis=0).item()
        total_area += area
        total_points += points
        if points / area > max_density:
            max_density = points / area
    print("density",density)
    print("density_p",density_n)
    print("density_n",density_p)
    print("density_eff",total_points/total_area)
    print("density_eff_max",max_density)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)

    args = parser.parse_args()

    result_path = 'result_lookup'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'test'
    item = args.item
    time_stamp_start = args.end - 50000
    time_stamp_end = args.end
    data_path = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_td.dat")
    bbox_file = os.path.join(final_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size, _ = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    #print(target)
    f_event = PSEELoader(event_file)
    f_event.seek_time(time_stamp_start)
    events = f_event.load_delta_t(time_stamp_end - time_stamp_start)
    x,y,t,p = events['x'], events['y'], events['t'], events['p']
    events = np.stack([x.astype(int), y.astype(int), t, p], axis=-1)
    print(len(events))
    volume = generate_event_volume(events,(240,304),5)
    visualizeVolume(volume,dat_bbox,item,result_path,time_stamp_start,time_stamp_end)