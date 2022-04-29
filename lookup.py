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

import time

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x

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
    gt = gt[gt['t']==time_stamp_end]
    # for i in range(0,volume.shape[0],2):
    #     c_p = volume[i+1]
    #     c_p = 127 * c_p / (np.percentile(c_p,0.99)+1e-8)
    #     c_p = np.where(c_p>127, 127, c_p)
    #     c_n = volume[i]
    #     c_n = 127 * c_n / (np.percentile(c_p,0.99)+1e-8)
    #     c_n = np.where(c_n>127, 127, c_n)
    #     img_s = img + c_p[:,:,None].astype(np.uint8) - c_n[:,:,None].astype(np.uint8)
    #     draw_bboxes(img_s,gt,0,LABELMAP)
    #     path_t = os.path.join(path,filename+"_{0}".format(int(time_stamp_end)))
    #     if not(os.path.exists(path_t)):
    #         os.mkdir(path_t)
    #     cv2.imwrite(os.path.join(path_t,'{0}.png'.format(i)),img_s)
    c_p = volume[5:]
    #c_p = volume[-1:]
    c_p = c_p.sum(axis=0)
    c_n = volume[:5]
    c_n = c_n.sum(axis=0)
    c_p = np.where(c_p>c_n,c_p,0)
    c_p = c_p/5
    c_p = np.where(c_p>1.0,127.0,c_p*127)
    c_n = np.where(c_n>c_p,c_n,0)
    c_n = c_n/5
    c_n = np.where(c_n>1.0,-127.0,-c_n*127)
    c_map = c_p+c_n
    #c_map = np.where(c_n>0,-127,0)
    #c_map = np.where(c_p>0,127,0)
    img_s = img + c_map.astype(np.uint8)[:,:,None]
    #draw_bboxes(img_s,gt,0,LABELMAP)
    path_t = os.path.join(path,filename+"_{0}_window{1}_neg.png".format(int(time_stamp_end),time_stamp_end-time_stamp_start))
    cv2.imwrite(path_t,img_s)
    points_in_view = np.sum(np.sum(np.sum(volume>0,axis=0),axis=0),axis=0)
    density = points_in_view/(volume.shape[0]*volume.shape[1]*volume.shape[2])
    density_p = 2*np.sum(np.sum(np.sum(volume[1::2]>0,axis=0),axis=0),axis=0)/(volume.shape[0]*volume.shape[1]*volume.shape[2])
    density_n = 2*np.sum(np.sum(np.sum(volume[0::2]>0,axis=0),axis=0),axis=0)/(volume.shape[0]*volume.shape[1]*volume.shape[2])
    bbox_mask = np.zeros([volume.shape[0],volume.shape[1],volume.shape[2]],dtype=bool)
    gt_trans = gt
    max_density = 0
    min_density = 1.0
    for j in range(len(gt_trans)):
        x, y, w, h = gt_trans['x'][j], gt_trans['y'][j], gt_trans['w'][j], gt_trans['h'][j]
        x = np.where(x<0, 0, x)
        y = np.where(y<0, 0, y)
        w = np.where(x + w > volume.shape[2], volume.shape[2] - x, w)
        h = np.where(y + h > volume.shape[1], volume.shape[1] - y, h)
        area = w * h * volume.shape[0]
        if (area <= 0) or (w<0) or (h<0):
            continue
        points = np.sum(np.sum(np.sum(volume[:,int(y):int(y+h),int(x):int(x+w)]>0,axis=0),axis=0),axis=0)
        bbox_mask[:,int(y):int(y+h),int(x):int(x+w)] = True
        if points / area > max_density:
            max_density = points / area
        if points / area < min_density:
            min_density = points / area
    density_eff = np.sum(np.sum(np.sum(volume[bbox_mask]>0,axis=0),axis=0),axis=0)/(np.sum(np.sum(np.sum(bbox_mask,axis=0),axis=0),axis=0))
    density_uneff = np.sum(np.sum(np.sum(volume[~bbox_mask]>0,axis=0),axis=0),axis=0)/(np.sum(np.sum(np.sum(~bbox_mask,axis=0),axis=0),axis=0))
    print("density",density)
    print("density_p",density_n)
    print("density_n",density_p)
    print("density_eff",density_eff)
    print("density_uneff",density_uneff)
    print("density_eff_max",max_density)
    print("density_eff_min",min_density)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)
    parser.add_argument('-window', type=int, default=50000)

    args = parser.parse_args()

    result_path = 'result_lookup'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'test'
    item = args.item
    time_stamp_start = args.end - args.window
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
    volume = generate_event_volume(events,(240,304),5)
    volume = torch.from_numpy(volume).cuda()
    median_filter = MedianPool2d(3,1,same=True).cuda()
    start = time.time()
    volume = median_filter(volume)
    torch.cuda.synchronize()
    print("elapse",time.time()-start)
    visualizeVolume(volume,dat_bbox,item,result_path,time_stamp_start,time_stamp_end)