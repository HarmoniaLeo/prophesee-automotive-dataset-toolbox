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
    parser.add_argument('-optical_min_start', type=float)
    parser.add_argument('-optical_min_end', type=float)
    parser.add_argument('-optical_max_start', type=float)
    parser.add_argument('-optical_max_end', type=float)
    parser.add_argument('-basic_root', type=str)
    parser.add_argument('-taf_root', type=str)
    parser.add_argument('-tol', type = int, default=4999)

    args = parser.parse_args()
    mode = "test"

    if args.dataset == "gen1":
        result_path = "/home/lbd/100-fps-event-det/" + args.basic_root + "/summarise.npz"
        taf_result_path = "/home/lbd/100-fps-event-det/" + args.taf_root + "/summarise.npz"
        shape = [240,304]
        filter_boxes = filter_boxes_gen1
        classes = ['Car', "Pedestrian"]
    else:
        result_path = "/home/liubingde/100-fps-event-det/" + args.basic_root + "/summarise.npz"
        taf_result_path = "/home/liubingde/100-fps-event-det/" + args.taf_root + "/summarise.npz"
        shape = [720,1280]
        filter_boxes = filter_boxes_large
        classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']
        
    bbox_file = result_path
    f_bbox = np.load(bbox_file)
    dts = f_bbox["dts"]
    file_names_dt = f_bbox["file_names"]
    densitys_dt = f_bbox["densitys"]

    bbox_file = taf_result_path
    f_bbox = np.load(bbox_file)
    dts_taf = f_bbox["dts"]
    file_names_dt_taf = f_bbox["file_names"]
    densitys_dt_taf = f_bbox["densitys"]

    result_path = "statistics_result"
    bbox_file = os.path.join(result_path,"gt_"+args.dataset+".npz")
    f_bbox = np.load(bbox_file)
    gts = f_bbox["gts"]
    file_names_gt = f_bbox["file_names"]
    densitys_gt = f_bbox["densitys"]

    results = []
    results_taf = []
    files = []
    timestamps = []

    for file_name in np.unique(file_names_gt):
        dts_file = dts[file_names_dt == file_name]
        densitys_dt_file = densitys_dt[file_names_dt == file_name]
        dts_taf_file = dts_taf[file_names_dt_taf == file_name]
        densitys_dt_taf_file = densitys_dt_taf[file_names_dt_taf == file_name]
        gts_file = gts[file_names_gt == file_name]
        densitys_gt_file = densitys_gt[file_names_gt == file_name]
        for time_stamp in np.unique(gts_file[:,0]):
            min_density = np.min(densitys_gt_file[gts_file[:,0] == time_stamp])
            max_density = np.max(densitys_gt_file[gts_file[:,0] == time_stamp])
            if (min_density >= args.optical_min_start) & (min_density < args.optical_min_end) & (max_density >= args.optical_max_start) & (max_density < args.optical_max_end):
                files.append(file_name)
                timestamps.append(time_stamp) 
                gt_boxes_list = map(filter_boxes, [gts_file[gts_file[:,0] == time_stamp]])
                result_boxes_list = map(filter_boxes, [dts_file[(dts_file[:,0] >= time_stamp - args.tol) & (dts_file[:,0] <= time_stamp + args.tol)]])
                result_boxes_list_taf = map(filter_boxes, [dts_taf_file[(dts_file[:,0] >= time_stamp - args.tol) & (dts_file[:,0] <= time_stamp + args.tol)]])
                gt_boxes_list1 = []
                result_boxes_list1 = []
                result_boxes_list1_taf = []
                for l1,l2,l3 in zip(gt_boxes_list,result_boxes_list,result_boxes_list1_taf):
                    if len(l1) > 0:
                        gt_boxes_list1.append(l1)
                        if len(l2) == 0:
                            result_boxes_list1.append(np.array([[l1[0,0],0,0,0,0,0,0,0]]))
                        else:
                            result_boxes_list1.append(l2)
                        if len(l3) == 0:
                            result_boxes_list1_taf.append(np.array([[l1[0,0],0,0,0,0,0,0,0]]))
                        else:
                            result_boxes_list1_taf.append(l3)
                
                eval_results = evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = args.tol, classes=classes,height=filter_boxes[0],width=filter_boxes[1])
                results.append(eval_results[0])
                eval_results = evaluate_detection(gt_boxes_list1, result_boxes_list1_taf, time_tol = args.tol, classes=classes,height=filter_boxes[0],width=filter_boxes[1])
                results_taf.append(eval_results[0])
                
    indices = np.argsort(np.array(results_taf)-np.array(results))
    for ind in indices:
        print(files[ind],timestamps[ind],results_taf[ind],timestamps[ind])