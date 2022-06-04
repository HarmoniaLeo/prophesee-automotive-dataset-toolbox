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
        percentiles = [0.009751069680872804, 0.017454583215420916, 0.019626774417120713, 0.0220462926022213, 0.02402894504496425, 0.026750427499481293, 0.031298709160567005, 0.03780696540140833, 0.04959067215527764, 0.06663794818976748, 0.09032619747330549, 0.1138681130249937, 0.14646434373294823, 0.18555400454792828, 0.23855403506488104, 0.3240646265432454, 0.42880768428242144, 0.5726358303437851, 0.84624335912686, 1.4038680516432203, 1000]
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
        for file_name in np.unique(file_names_dt):
            dts_file = dts[file_names_dt == file_name]
            densitys_dt_file = densitys_dt[file_names_dt == file_name]
            gts_file = gts[file_names_gt == file_name]
            densitys_gt_file = densitys_gt[file_names_gt == file_name]
            for time_stamp in np.unique(dts_file[:,0]):
                if time_stamp < 500000:
                    continue
                #dts_to_eval = dts_file[(dts_file[:,0] >= time_stamp - args.tol) & (dts_file[:,0] <= time_stamp + args.tol) & (densitys_dt_file >= percentiles[i]) & (densitys_dt_file < percentiles[i+1])]
                dts_to_eval = dts_file[(dts_file[:,0] >= time_stamp - args.tol) & (dts_file[:,0] <= time_stamp + args.tol)]
                #gts_to_eval = gts_file[(gts_file[:,0] == time_stamp) & (densitys_gt_file >= percentiles[i]) & (densitys_gt_file < percentiles[i+1])]
                gts_to_eval = gts_file[(gts_file[:,0] >= time_stamp - args.tol) & (gts_file[:,0] <= time_stamp + args.tol)]
                print(gts_to_eval - dts_to_eval)
                dt.append(dts_to_eval)
                gt.append(gts_to_eval)
                #raise Exception("break")
    
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
        
        evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = args.tol, classes=classes,height=shape[0],width=shape[1])
        break
    # print([(percentiles[i] + percentiles[i+1])/2 for i in range(0,len(percentiles)-1)])
    # print(results)