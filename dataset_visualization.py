"""
small executable to show the content of the Prophesee dataset
Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import cv2
import argparse
from glob import glob
from os import listdir
import os

from src.visualize import vis_utils as vis
import tqdm

from src.io.psee_loader import PSEELoader


def play_files_parallel(td_files, labels=None, delta_t=250000, skip=0):
    """
    Plays simultaneously files and their boxes in a rectangular format.
    """
    # open the video object for the input files
    td_files = listdir("/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0/test")
    td_files = [file for file in td_files if file[-3:] == 'dat']
    td_files = ["17-04-11_15-13-23_244500000_304500000_td.dat"]

    # optionally skip n minutes in all videos

    # preallocate a grid to display the images
    

    #cv2.namedWindow('out', cv2.WINDOW_NORMAL)

    # while all videos have something to read
    # load events and boxes from all files
    pbar = tqdm.tqdm(total=len(td_files), unit='File', unit_scale=True)
    for i,td_file in enumerate(td_files):
        out_path = os.path.join("buf2",td_file)
        try:
            os.makedirs(out_path)
        except Exception:
            pbar.update(1)
            continue
        #print(out_path)
        td_file = os.path.join("/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0/test",td_file)
        #print(td_file)
        video = PSEELoader(td_file)
        #box_video = PSEELoader(glob(td_file.split('_td.dat')[0] +  '*.npy')[0])
        box_events = PSEELoader(glob(td_file.split('_td.dat')[0] +  '*.npy')[0])
        end_ts = box_events.total_time()
        box_events = box_events.load_delta_t(end_ts)
        height, width = video.get_size()
        video.seek_time(skip)
        #box_video.seek_time(skip)
        labelmap = vis.LABELMAP if height == 240 else vis.LABELMAP_LARGE
        last_time = skip
        count = 0
        for t in np.unique(box_events['t']):
        #while not video.done:
            #event = video.load_n_events(delta_t)
            video.seek_time(t)
            event = video.load_delta_t(delta_t)
            while len(event) == 0:
                event = video.load_delta_t(delta_t)
            #while not(len(event)>25000):
            #    event1 = video.load_delta_t(delta_t)
            #    event = np.concatenate([event,event1])
            ts = event["ts"]
            box_event = box_events[(box_events['t']>=t)&(box_events['t']<=t+delta_t)]
                
            #ts = event["ts"]
            #t_max = np.max(ts)
            #t_min = np.min(ts)
            #box_video.seek_time(t_min)
            #if t_max-t_min<1:
            #    box_event = box_video.load_delta_t(1)
            #else:
            #    box_event = box_video.load_delta_t(t_max-t_min)
            #box_event = box_video.load_delta_t(delta_t)
            
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame = vis.make_binary_histo(event, img=frame, width=width, height=height)
            vis.draw_bboxes(frame, box_event, labelmap=labelmap)
            if len(box_event)>0:
                cv2.imwrite(os.path.join(out_path,'{0}-t.png'.format(count)),frame)
            else:
                cv2.imwrite(os.path.join(out_path,'{0}.png'.format(count)),frame)

            video.seek_time(t)
            event = video.load_n_events(delta_t)
            #while not(len(event)>25000):
            #    event1 = video.load_delta_t(delta_t)
            #    event = np.concatenate([event,event1])
            ts = event["ts"]
            last_time = np.max(ts)
            box_event = box_events[(box_events['t']>=t)&(box_events['t']<=t+delta_t)]
                
            #ts = event["ts"]
            #t_max = np.max(ts)
            #t_min = np.min(ts)
            #box_video.seek_time(t_min)
            #if t_max-t_min<1:
            #    box_event = box_video.load_delta_t(1)
            #else:
            #    box_event = box_video.load_delta_t(t_max-t_min)
            #box_event = box_video.load_delta_t(delta_t)
            
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame = vis.make_binary_histo(event, img=frame, width=width, height=height)
            vis.draw_bboxes(frame, box_event, labelmap=labelmap)
            if len(box_event)>0:
                cv2.imwrite(os.path.join(out_path,'{0}-n.png'.format(count)),frame)
            else:
                cv2.imwrite(os.path.join(out_path,'{0}.png'.format(count)),frame)
            count+=1
        pbar.update(1)
    pbar.close()


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('records', nargs="+",
                        help='input event files, annotation files are expected to be in the same folder')
    parser.add_argument('-s', '--skip', default=0, type=int, help="skip the first n microseconds")
    parser.add_argument('-d', '--delta_t', default=50000, type=int, help="load files by delta_t in microseconds")

    return parser.parse_args()


if __name__ == '__main__':
    ARGS = parse_args()
    play_files_parallel(ARGS.records, skip=ARGS.skip, delta_t=ARGS.delta_t)
