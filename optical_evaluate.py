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
        percentiles = [0, 0.002708115195855499, 0.026879342272877697, 0.06864213943481444, 0.23081630468368539]
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
        for file_name in np.unique(file_names_gt):
            dts_file = dts[file_names_dt == file_name]
            densitys_dt_file = densitys_dt[file_names_dt == file_name]
            gts_file = gts[file_names_gt == file_name]
            densitys_gt_file = densitys_gt[file_names_gt == file_name]
            for time_stamp in np.unique(gts_file[:,0]):
                dts_to_eval = dts_file[(dts_file[:,0] >= time_stamp - args.tol) & (dts_file[:,0] <= time_stamp + args.tol) & (densitys_dt_file >= percentiles[i]) & (densitys_dt_file < percentiles[i+1])]
                gts_to_eval = gts_file[(gts_file[:,0] == time_stamp) & (densitys_gt_file >= percentiles[i]) & (densitys_gt_file < percentiles[i+1])]
                dt.append(dts_to_eval)
                gt.append(gts_to_eval)
    
        gt_boxes_list = map(filter_boxes, gt)
        result_boxes_list = map(filter_boxes, dt)
        gt_boxes_list1 = []
        result_boxes_list1 = []
        for l1,l2 in zip(gt_boxes_list,result_boxes_list):
            if len(l1) > 0:
                gt_boxes_list1.append(l1)
                if len(l2) == 0:
                    result_boxes_list1.append(np.array([[l1[0,0],0,0,0,0,0,0,0]]))
                else:
                    result_boxes_list1.append(l2)
        
        eval_results = evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = args.tol, classes=classes,height=filter_boxes[0],width=filter_boxes[1])
        results.append(eval_results[0])
    print([(percentiles[i] + percentiles[i+1])/2 for i in range(0,len(percentiles)-1)])
    print(results)