import os
import cv2
import numpy as np
from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_style("darkgrid")


def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
    #TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
    # TVL1 = cv2.DualTVL1OpticalFlow_create()
    TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32
 
    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0
 
    return flow

def cal_for_frames(volume1, volume2):
 
    prev = volume1
    #prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr = volume2
    flow = compute_TVL1(prev, curr)
 
    return flow
 
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

def save_flow(flow, gt,filename,flow_path,time_stamp_end):
    if not os.path.exists(os.path.join(flow_path, 'u')):
        os.mkdir(os.path.join(flow_path, 'u'))
    if not os.path.exists(os.path.join(flow_path, 'v')):
        os.mkdir(os.path.join(flow_path, 'v'))
    flow_u = flow[None, :, :, 0]
    draw_bboxes(flow_u,gt)
    flow_v = flow[None, :, :, 1]
    draw_bboxes(flow_v,gt)
    cv2.imwrite(os.path.join(flow_path,filename+"_end{0}_u.png".format(time_stamp_end)),flow_u)
    cv2.imwrite(os.path.join(flow_path,filename+"_end{0}_v.png".format(time_stamp_end)),flow_v)
 
def extract_flow(volume1, volume2, gt,filename,path,time_stamp_end):
    flow = cal_for_frames(volume1, volume2)
    save_flow(flow, gt,filename,path,time_stamp_end)

def generate_timesurface(events,shape,end_stamp):
    volume1, volume2 = np.zeros(shape), np.zeros(shape)
    for event in events:
        if event[2] < end_stamp - 10000:
            volume1[event[1]][event[0]] = event[2]
        volume2[event[1]][event[0]] = event[2]
    return volume1, volume2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-item', type=str)
    parser.add_argument('-end', type=int)

    args = parser.parse_args()

    result_path = 'optical_lookup'
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
    f_event.seek_event(end_count - 800000)
    events = f_event.load_n_events(800000)
    x,y,t,p = events['x'], events['y'], events['t'], events['p']
    events = np.stack([x.astype(int), y.astype(int), t, p], axis=-1)
    volume1, volume2 = generate_timesurface(events,(240,304),args.end)
    extract_flow(volume1, volume2, dat_bbox[(dat_bbox['t']==time_stamp_end)],item,result_path,time_stamp_end)