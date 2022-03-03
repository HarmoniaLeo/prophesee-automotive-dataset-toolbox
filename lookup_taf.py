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

def generate_event_volume(events,shape,time_start,time_end,past_volume=None):
    H, W = shape
    x, y, t, p = events.T

    ind = (t>=time_start)&(t<=time_end)
    x, y, t, p = x[ind], y[ind], t[ind], p[ind]

    x, y, p = x.astype(np.int), y.astype(np.int), p.astype(np.int)

    try:
        t_min = np.min(t)
        t_max = np.max(t)
        t = t.astype(np.float)
        t = (t-t_min)/(t_max-t_min+1e-8)

        t_star = t[:,None]
        
        xpos = x[p == 1]
        ypos = y[p == 1]
        adderpos = np.arange(0,2)[None,:]
        adderpos = 1 - np.abs(adderpos-t_star[p == 1])
        adderpos = np.where(adderpos>=0,adderpos,0)

        xneg = x[p == 0]
        yneg = y[p == 0]
        adderneg = np.arange(0,2)[None,:]
        adderneg = 1 - np.abs(adderneg-t_star[p == 0])
        adderneg = np.where(adderneg>=0,adderneg,0)

        img_pos = np.zeros((H * W , 2),dtype=float)
        np.add.at(img_pos, W * ypos + xpos, adderpos)
        img_neg = np.zeros((H * W , 2),dtype=float)
        np.add.at(img_neg, W * yneg + xneg, adderneg)
    except Exception:
        img_pos = np.zeros((H * W , 2),dtype=float)
        img_neg = np.zeros((H * W , 2),dtype=float)
    
    forward_pos = (img_pos[:,-1]==0)
    forward_neg = (img_neg[:,-1]==0)
    if not (past_volume is None):
        img_pos_old, img_neg_old, latest_pos, latest_neg, pos_ecd, neg_ecd = past_volume
        img_pos_old[:,-1] = np.where(latest_pos,img_pos_old[:,-1] + img_pos[:,0],img_pos_old[:,-1])
        img_neg_old[:,-1] = np.where(latest_neg,img_neg_old[:,-1] + img_neg[:,0],img_neg_old[:,-1])
        img_pos = np.concatenate([img_pos_old,img_pos[:,1:]],axis=1)[:,1:]
        pos_ecd = np.concatenate([pos_ecd,np.zeros_like(pos_ecd[:,-1:])],axis=1)[:,1:]
        img_neg = np.concatenate([img_neg_old,img_neg[:,1:]],axis=1)[:,1:]
        neg_ecd = np.concatenate([neg_ecd,np.zeros_like(neg_ecd[:,-1:])],axis=1)[:,1:]
        for i in range(1,img_pos.shape[1])[::-1]:
            img_pos[:,i] = np.where(forward_pos, img_pos[:,i-1],img_pos[:,i])
            pos_ecd[:,i] = np.where(forward_pos, pos_ecd[:,i-1]-1,pos_ecd[:,i])
            img_neg[:,i] = np.where(forward_neg, img_neg[:,i-1],img_neg[:,i])
            neg_ecd[:,i] = np.where(forward_neg, neg_ecd[:,i-1]-1,neg_ecd[:,i])
        img_pos[:,0] = np.where(forward_pos, 0, img_pos[:,0])
        pos_ecd[:,0] = np.where(forward_pos, -1e6, pos_ecd[:,0])
        img_neg[:,0] = np.where(forward_neg, 0, img_neg[:,0])
        neg_ecd[:,0] = np.where(forward_neg, -1e6, neg_ecd[:,0])
    else:
        pos_ecd = np.where(forward_pos, -1e6, 0)[:,None]
        neg_ecd = np.where(forward_neg, -1e6, 0)[:,None]
        img_neg, img_pos = img_neg[:,1:], img_pos[:,1:]

    histogram = np.concatenate([img_neg, img_pos], -1).reshape((H, W, 2)).transpose(2,0,1)
    ecd = np.concatenate([neg_ecd, pos_ecd], -1).reshape((H, W, 2)).transpose(2,0,1)

    return histogram, ecd, (img_pos, img_neg, ~forward_pos, ~forward_neg, pos_ecd, neg_ecd)

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

def visualizeVolume(volume,ecd,gt_i,filename,path,pct,time_stamp_start,time_stamp_end,i):
    for j in range(2):
        img_s = 255 * np.ones((volume.shape[1], volume.shape[2], 3), dtype=np.uint8)
        print(np.unique(ecd[j]))
        img = (90 * np.exp(ecd[j])).astype(np.uint8) + 89
        img_s[:,:,0] = img
        img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
        draw_bboxes(img_s,gt_i)
        path_t = os.path.join(path,filename+"_start{0}_end{1}".format(int(time_stamp_start),int(time_stamp_end)))
        if not(os.path.exists(path_t)):
            os.mkdir(path_t)
        cv2.imwrite(os.path.join(path_t,'{1}-{0}.png'.format(i,j)),img_s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-start', type=int)
    parser.add_argument('-end', type=int)
    parser.add_argument('-window', type=int, default=10000)
    parser.add_argument('-upper_thr', type=float, default=90)
    parser.add_argument('-per_time_bbox', type=bool, default=False)

    args = parser.parse_args()

    result_path = 'result_taf'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'test'
    item = args.item
    time_stamp_start = args.start
    time_stamp_end = args.end
    data_path = "/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    final_path = os.path.join(data_path,data_folder)
    event_file = os.path.join(final_path, item+"_td.dat")
    bbox_file = os.path.join(final_path, item+"_bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    #print(target)
    f_event = PSEELoader(event_file)
    f_event.seek_time(time_stamp_start)
    events = f_event.load_delta_t(time_stamp_end - time_stamp_start)
    x,y,t,p = events['x'], events['y'], events['t'], events['p']
    events = np.stack([x.astype(int), y.astype(int), t, p], axis=-1)
    print(len(events))
    past_volume = None
    for i,start in enumerate(range(time_stamp_start, time_stamp_end, args.window)):
        volume, ecd, past_volume = generate_event_volume(events,(240,304),start,start+args.window,past_volume)
        if args.per_time_bbox:
            gt_i = dat_bbox[(dat_bbox['t']>=start)&(dat_bbox['t']<=time_stamp_end)]
        else:
            gt_i = dat_bbox[(dat_bbox['t']>=time_stamp_start)&(dat_bbox['t']<=time_stamp_end)]
        visualizeVolume(volume,ecd,gt_i,item,result_path,args.upper_thr,time_stamp_start,time_stamp_end,i)