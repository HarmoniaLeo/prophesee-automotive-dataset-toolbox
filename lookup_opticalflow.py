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

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    UNKNOWN_FLOW_THRESH = 1e7
    SMALLFLOW = 0.0
    LARGEFLOW = 1e8

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    u = np.where(u>np.quantile(u,0.98),np.quantile(u,0.98),u)
    u = np.where(u<np.quantile(u,0.02),np.quantile(u,0.02),u)
    v = np.where(v>np.quantile(v,0.98),np.quantile(v,0.98),v)
    v = np.where(v<np.quantile(u,0.02),np.quantile(v,0.02),v)

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def compute_TVL1(prev, curr, bound=1):
    """Compute the TV-L1 optical flow."""
    TVL1=cv2.optflow.DualTVL1OpticalFlow_create()
    #TVL1 = cv2.DualTVL1OpticalFlow_create()
    #TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    # assert flow.dtype == np.float32
    
    # flow = np.sqrt(flow[:,:,:1] ** 2 + flow[:,:,1:2] ** 2)
    # flow = (flow + bound) * (255.0 / (2 * bound))
    # flow = np.round(flow).astype(int)
    # flow[flow >= 255] = 255
    # flow[flow <= 0] = 0
 
    return flow

def cal_for_frames(volume1, volume2):
 
    prev = volume1
    #prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr = volume2
    flow = compute_TVL1(prev, curr)
 
    return flow
 
LABELMAP = ["car", "pedestrian"]

def draw_bboxes(img, flow, boxes, dt = 0, labelmap=LABELMAP):
    """
    draw bboxes in the image img
    """
    colors = cv2.applyColorMap(np.arange(0, 255).astype(np.uint8), cv2.COLORMAP_HSV)
    colors = [tuple(*item) for item in colors.tolist()]

    for i in range(boxes.shape[0]):
        pt1 = (int(boxes[i][1]), int(boxes[i][2]))
        size = (int(boxes[i][3]), int(boxes[i][4]))
        pt2 = (pt1[0] + size[0], pt1[1] + size[1])
        print(np.sum(np.sqrt(flow[pt1[1]:pt2[1],pt1[0]:pt2[0]]**2))/size[0]/size[1])
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
    if not os.path.exists(flow_path):
        os.mkdir(flow_path)
    # flow_u = flow[:, :, :1]
    # draw_bboxes(flow_u,gt)
    # flow_v = flow[:, :, 1:2]
    # draw_bboxes(flow_v,gt)
    #cv2.imwrite(os.path.join(flow_path,filename+"_end{0}_u.png".format(time_stamp_end)),flow_u)
    flow_img = 255 - flow_to_image(flow)
    draw_bboxes(flow_img, flow,gt)
    cv2.imwrite(os.path.join(flow_path,filename+"_end{0}.png".format(time_stamp_end)),flow_img)
    #cv2.imwrite(os.path.join(flow_path,filename+"_end{0}_v.png".format(time_stamp_end)),flow_v)

def visualize_timesuface(volume1, volume2, gt,filename,flow_path,time_stamp_end):
    if not os.path.exists(flow_path):
        os.mkdir(flow_path)

    cv2.imwrite(os.path.join(flow_path,filename+"_end{0}_u.png".format(time_stamp_end)),volume1)
    cv2.imwrite(os.path.join(flow_path,filename+"_end{0}_v.png".format(time_stamp_end)),volume2)
 
def extract_flow(volume1, volume2, gt,filename,path,time_stamp_end):
    flow = cal_for_frames(volume1, volume2)
    save_flow(flow, gt,filename,path,time_stamp_end)
    #visualize_timesuface(volume1, volume2, gt,filename,path,time_stamp_end)

# def generate_timesurface(events,shape,end_stamp):
#     volume1, volume2 = np.zeros(shape), np.zeros(shape)
#     end_stamp = events[:,2].max()
#     start_stamp = events[:,2].min()
#     print(end_stamp)
#     for event in events:
#         if event[2] < end_stamp - 50000:
#             volume1[event[1]][event[0]] = event[2]
#         volume2[event[1]][event[0]] = event[2]
#     volume1 = volume1 - start_stamp
#     volume2 = volume2 - start_stamp - 50000
#     volume1 = volume1 / (end_stamp - 50000 - start_stamp) * 255
#     volume2 = volume2 / (end_stamp - 50000 - start_stamp) * 255
#     # volume1 = volume1 - events[:,2].max() + 50000
#     # volume2 = volume2 - events[:,2].max() + 40000
#     # volume1 = volume1 / 50000 * 255
#     # volume2 = volume2 / 50000 * 255
#     volume1 = np.where(volume1<0, 0, volume1)
#     volume2 = np.where(volume2<0, 0, volume2)
#     return volume1.astype(np.uint8), volume2.astype(np.uint8)

def generate_timesurface(events,shape,start_stamp,end_stamp,buffer):
    if not (buffer is None):
        volume1 = buffer
        volume2 = buffer
    else:
        volume1, volume2 = np.zeros(shape), np.zeros(shape)
    # end_stamp = events[:,2].max()
    # start_stamp = events[:,2].min()
    for event in events:
        if event[2] < end_stamp - 50000:
            volume1[int(event[1])][int(event[0])] = event[2]
        volume2[int(event[1])][int(event[0])] = event[2]
    buffer = volume2
    volume2 = volume2 - 50000
    end_stamp = end_stamp - 50000
    volume2 = np.where(volume2 > start_stamp, volume2, start_stamp)
    volume1 = (volume1 - start_stamp) / (end_stamp - start_stamp) * 255
    volume2 = (volume2 - start_stamp) / (end_stamp - start_stamp) * 255
    # volume1 = volume1 - events[:,2].max() + 50000
    # volume2 = volume2 - events[:,2].max() + 40000
    # volume1 = volume1 / 50000 * 255
    # volume2 = volume2 / 50000 * 255
    # volume1 = np.where(volume1<0, 0, volume1)
    # volume2 = np.where(volume2<0, 0, volume2)
    return volume1.astype(np.uint8), volume2.astype(np.uint8), buffer


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
    #end_count = f_event.seek_time(time_stamp_end)
    #f_event.seek_event(end_count - 200000)
    #time_stamp_start = f_event.current_time
    time_stamp_start = time_stamp_end - 500000
    end_count = f_event.seek_time(time_stamp_end)
    events = f_event.load_delta_t(time_stamp_end-time_stamp_start)
    x,y,t,p = events['x'], events['y'], events['t'], events['p']
    events = np.stack([x.astype(int), y.astype(int), t, p], axis=-1)
    volume1, volume2 = generate_timesurface(events,(240,304),time_stamp_start,time_stamp_end)
    extract_flow(volume1, volume2, dat_bbox[(dat_bbox['t']==time_stamp_end)],item,result_path,time_stamp_end)
    visualize_timesuface(volume1,volume2,dat_bbox[(dat_bbox['t']==time_stamp_end)],item,result_path,time_stamp_end)