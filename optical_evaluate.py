from cgitb import small
from fileinput import filename
import numpy as np
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import argparse
from src.io.box_filtering import filter_boxes_gen1, filter_boxes_large
from src.metrics.coco_eval import evaluate_detection

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-exp_name', type=str)
    parser.add_argument('-tol', type = int, default=4999)

    args = parser.parse_args()
    mode = "test"

    if args.dataset == "gen1":
        result_path = "/home/lbd/100-fps-event-det/log/" + args.exp_name + "/summarise.npz"
        shape = [240,304]
        filter_boxes = filter_boxes_gen1
        classes = ['Car', "Pedestrian"]
        #percentiles = [0.0, 0.022496978317470943, 0.03584141107151823, 0.06461365563824012, 0.09765842901836184, 0.13363889435484622, 0.17335709874840652, 0.21913896857877602, 0.2795803473626733, 0.34990456407260045, 0.44142171223006244, 0.557804203925751, 0.7039533807928522, 0.8876761367290176, 1.1264612928414415, 1.4447705627987673, 1.8662489229530281, 2.4500051802140845, 3.242211733282628, 4.436053050035189, 1000]
        percentiles = [0.0, 0.04511805096746158, 0.19679771615254632, 0.5256959230971759, 1.4566068453489602, 1000]
    else:
        result_path = "/home/liubingde/100-fps-event-det/log/" + args.exp_name + "/summarise.npz"
        shape = [720,1280]
        filter_boxes = filter_boxes_large
        classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']
        
    bbox_file = result_path
    f_bbox = np.load(bbox_file)
    dts = f_bbox["dts"]
    file_names_dt = f_bbox["file_names"]
    densitys_dt = f_bbox["densitys"]

    result_path = "statistics_result"
    bbox_file = os.path.join(result_path,"gt_"+args.dataset+".npz")
    f_bbox = np.load(bbox_file)
    gts = f_bbox["gts"]
    file_names_gt = f_bbox["file_names"]
    densitys_gt = f_bbox["densitys"]

    results = []

    for i in range(0,len(percentiles)-1):
        print(i,percentiles[i],percentiles[i+1])
        dt = []
        gt = []

        for i_file, file_name in enumerate(np.unique(file_names_gt)):

            #dt_bbox = dts[(file_names_dt == file_name)&(densitys_dt >= percentiles[i])&(densitys_dt < percentiles[i+1])]
            #gt_bbox = gts[(file_names_gt == file_name)&(densitys_gt >= percentiles[i])&(densitys_gt < percentiles[i+1])]
            dt_bbox = dts[(file_names_dt == file_name)]
            gt_bbox = gts[(file_names_gt == file_name)]

            dt.append(dt_bbox)
            gt.append(gt_bbox)

        gt_boxes_list = map(filter_boxes, gt)
        result_boxes_list = map(filter_boxes, dt)
        # gt_boxes_list1 = []
        # result_boxes_list1 = []
        # for l1,l2 in zip(gt_boxes_list,result_boxes_list):
        #     if len(l1) > 0:
        #         gt_boxes_list1.append(l1)
        #         if len(l2) == 0:
        #             result_boxes_list1.append(np.array([[l1[0,0],0,0,0,0,0,0,0]]))
        #         else:
        #             result_boxes_list1.append(l2)
        
        evaluate_detection(gt_boxes_list, result_boxes_list, time_tol = args.tol, classes=classes,height=shape[0],width=shape[1])
        break
    # print([(percentiles[i] + percentiles[i+1])/2 for i in range(0,len(percentiles)-1)])
    # print(results)