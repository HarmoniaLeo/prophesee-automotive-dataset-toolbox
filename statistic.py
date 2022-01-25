import numpy as np
from src.io.psee_loader import PSEELoader
from src.io import npy_events_tools
import os

def make_print_to_file(path='./'):
    '''
    path， it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime
 
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.log = open(os.path.join(path, filename), "a", encoding='utf8',)
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
 
 
    fileName = datetime.datetime.now().strftime('day'+'%Y_%m_%d')
    sys.stdout = Logger(fileName + '.log', path=path)
 
    #############################################################
    # 这里输出之后的所有的输出的print 内容即将写入日志
    #############################################################
    print(fileName.center(60,'*'))

time_window = 200000

def inv_statistic(data_path,item):
    bbox_file = os.path.join(data_path, item[:-6]+"bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)
    unique_inv = np.unique(unique_ts[1:] - unique_ts[:-1])
    if np.sum(np.where(unique_inv<=time_window)):
        print(item,unique_inv)

def full_statistic(data_path,item):
    event_file = os.path.join(data_path, item)
    bbox_file = os.path.join(data_path, item[:-6]+"bbox.npy")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()
    unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)
    f_event = PSEELoader(event_file)
    for bbox_count,unique_time in enumerate(unique_ts):
        start_time = unique_time - time_window
        if start_time >= 0:
            f_event.seek_time(start_time)
            events = f_event.load_delta_t(time_window)
            H, W = 304, 240
            x, y, t, p = events.T
            x = x.astype(np.int)
            y = y.astype(np.int)

            img_pos = np.zeros((H * W,), dtype="float32")
            img_neg = np.zeros((H * W,), dtype="float32")

            np.add.at(img_pos, x[p == 1] + W * y[p == 1], 0.05)
            np.add.at(img_neg, x[p == -1] + W * y[p == -1], 0.05)

            img_pos = np.where(img_pos>0,1,0)
            img_neg = np.where(img_neg>0,1,0)
            print(event_file,unique_time,np.sum(img_pos)/len(img_pos),np.sum(img_neg)/len(img_neg))

result_path = 'result_full'
make_print_to_file(path=result_path)
data_folders = ["train","test","val"]
data_path="/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
for data_folder in data_folders:
    final_path = os.path.join(data_path,data_folder)
    files = os.listdir(final_path)
    for item in files:
        lastStamp=0
        if os.path.isfile(os.path.join(final_path, item)):
            if item.endswith(".dat"):
                full_statistic(final_path,item)
                
                '''video = PSEELoader(os.path.join(data_path, item[:-8]+"td.dat"))
                while not video.done:
                    events = video.load_delta_t(10000)
                    boxes = gt.load_delta_t(10000)
                    if events.shape[0]>min_point:
                        events_list.append((os.path.join(data_path, item[:-8]+"td.dat"),np.min(events["t"]),np.max(events["t"])))
                        #print("event:",(os.path.join(data_path, item[:-8]+"td.dat"),np.min(events["t"]),np.max(events["t"])))
                    if boxes.shape[0]>0:
                        print(boxes.shape[0])
                        while len(boxes_list)<len(events_list):
                            boxes_list.append((os.path.join(data_path, item),np.min(boxes["t"])))
                            #print("boxes:",(os.path.join(data_path, item),np.min(boxes["t"]),np.max(boxes["t"])))
                    #print("start: ",np.min(events["t"]),np.max(events["t"]))
                    #print(events.shape)
                    #print(boxes.shape)
    #print(np.mean(count),np.max(count))'''
            