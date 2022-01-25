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

make_print_to_file(path='result_inv')
data_path="/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0/test"
files = os.listdir(data_path)
min_point=5000
events_list=[]
boxes_list=[]
boxes_count=[]
unique_invs = []
for item in files:
    lastStamp=0
    if os.path.isfile(os.path.join(data_path, item)):
        if item.endswith(".npy"):
            #gt = PSEELoader(os.path.join(data_path, item))
            bbox_file = os.path.join(data_path, item)
            try:
                f_bbox = open(bbox_file, "rb")
                start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
                dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
                f_bbox.close()
            except Exception:
                break
            unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)
            unique_inv = np.unique(unique_ts[1:] - unique_ts[:-1])
            if np.sum(np.where(unique_inv<=200000)):
                print(item,unique_inv)
            
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
            