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

def generate_tore(events,shape):
    q = np.zeros(shape)
    c = np.zeros(shape)
    K = 10
    for event in events:
        if event[3] == 1:
            if c[event[1]][event[0]] == 0:
                q[event[1]][event[0]] = event[2]
            c[event[1]][event[0]] += 1
            if c[event[1]][event[0]] == K:
                q[event[1]][event[0]] = event[2] - q[event[1]][event[0]]
    return q[c>=30]

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

def visualizeVolume(volume,gt,filename,path,time_stamp_end):
    img_s = 255 * np.ones((volume.shape[0], volume.shape[1], 3), dtype=np.uint8)
    print(volume.max(),np.quantile(volume, 0.95),volume.min(),np.quantile(volume, 0.05))
    # quant = np.quantile(volume, 0.95)
    # quant = 2
    min = volume.min()
    #volume = np.where(volume>5,5,volume)
    volume = (volume-min)/volume.max()
    tar = volume / volume.max()
    #tar = np.where(tar * 10 > 1, 1, tar)
    img_0 = (60 * tar).astype(np.uint8) + 119
    #img_1 = (255 * tar).astype(np.uint8)
    #img_2 = (255 * tar).astype(np.uint8)
    img_s[:,:,0] = img_0
    #img_s[:,:,1] = img_1
    #img_s[:,:,2] = img_2
    img_s = cv2.cvtColor(img_s, cv2.COLOR_HSV2BGR)
    draw_bboxes(img_s,gt)
    path_t = os.path.join(path,filename+"_end{0}.png".format(int(time_stamp_end)))
    cv2.imwrite(path_t,img_s)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)

    args = parser.parse_args()

    result_path = 'tore_lookup'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    data_folder = 'test'
    item = args.item
    #time_stamp_start = args.end - 50000
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
    end_count = f_event.seek_time(args.end)
    f_event.seek_event(end_count - 200000)
    #f_event.seek_event(0)
    events = f_event.load_n_events(200000)
    x,y,t,p = events['x'], events['y'], events['t'], events['p']
    events = np.stack([x.astype(int), y.astype(int), t, p], axis=-1)
    volume = generate_tore(events,(240,304))
    #volume = np.where(np.log1p(time_stamp_end - volume)>np.log1p(5000000), np.log1p(5000000), np.log1p(time_stamp_end - volume))
    print(volume.max(),volume.min(),volume.mean())
    #visualizeVolume(volume,dat_bbox[(dat_bbox['t']==time_stamp_end)],item,result_path,time_stamp_end)