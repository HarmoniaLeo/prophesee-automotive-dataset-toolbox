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
        result_path = "/home/lbd/100-fps-event-det/log/" + args.exp_name + "/summarise_stats.npz"
        shape = [240,304]
        filter_boxes = filter_boxes_gen1
        classes = ['Car', "Pedestrian"]
        #percentiles = [0.0, 0.022387557139397742, 0.035573737179988865, 0.06407627710119064, 0.09700126236027522, 0.1331314125494094, 0.17299334115956197, 0.21912188316533107, 0.2798854433302963, 0.35095609586053406, 0.4424646250749726, 0.560056164172371, 0.708670571285379, 0.8961528996548985, 1.1398348121349988, 1.4632983195825662, 1.8991032480445367, 2.5114790349998435, 3.328938628409255, 4.577429880336184]
        #percentiles = [0.0, 0.022090551140021526, 0.03307005296535371, 0.05671614855808474, 0.08410216123034962, 0.11406982861849572, 0.14831617911666944, 0.18462005594938136, 0.22870882848611626, 0.2853853858727559, 0.3554077913283515, 0.44819659632660175, 0.5715129938105777, 0.7367699321420207, 0.9613864329222702, 1.2917300269541516, 1.7416549267996466, 2.4035640687750264, 3.302250419545379, 4.655760829519853, 1000]
        #percentiles = [0.0, 0.03307005296535371, 0.08410216123034962, 0.14831617911666944, 0.22870882848611626, 0.3554077913283515, 0.5715129938105777, 0.9613864329222702, 1.7416549267996466, 3.302250419545379, 1000]
        #percentiles = [0.0, 0.08410216123034962, 0.22870882848611626, 0.5715129938105777, 1.7416549267996466, 1000]
        percentiles = [0.0, 0.022387557139397742, 0.035573737179988865, 0.06407627710119064, 0.09700126236027522, 0.1331314125494094, 0.17299334115956197, 0.21912188316533107, 0.2798854433302963, 0.35095609586053406, 0.4424646250749726, 0.560056164172371, 0.708670571285379, 0.8961528996548985, 1.1398348121349988, 1.4632983195825662, 1.8991032480445367, 2.5114790349998435, 3.328938628409255, 4.577429880336184]
    else:
        result_path = "/home/liubingde/100-fps-event-det/log/" + args.exp_name + "/summarise_stats.npz"
        shape = [720,1280]
        filter_boxes = filter_boxes_large
        classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']
        #percentiles = [0.0, 0.13798373765648486, 0.6976278461290158, 2.025066711356914, 5.278082102864997, 1000]
        #percentiles = [0.0, 0.0, 0.0, 0.02398205232206049, 0.13798373765648486, 0.26979497435076183, 0.3868838956735553, 0.5377039725305155, 0.6976278461290158, 0.9021997289279784, 1.1724968566047203, 1.542907747791367, 2.025066711356914, 2.620455828266434, 3.437184215541272, 4.26408848239479, 5.278082102864997, 6.375703482860002, 7.733929135075739, 9.711541416065378]
        percentiles = [0.0, 0.26979497435076183, 4.26408848239479, 1000]
        
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

            dt_bbox = dts[(file_names_dt == file_name)&(densitys_dt >= percentiles[i])&(densitys_dt < percentiles[i+1])]
            gt_bbox = gts[(file_names_gt == file_name)&(densitys_gt >= percentiles[i])&(densitys_gt < percentiles[i+1])]
            # dt_bbox = dts[(file_names_dt == file_name)]
            # gt_bbox = gts[(file_names_gt == file_name)]

            dt.append(dt_bbox)
            gt.append(gt_bbox)

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
        
        result = evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = args.tol, classes=classes,height=shape[0],width=shape[1])
        results.append(result[0])
    print([(percentiles[i] + percentiles[i+1])/2 for i in range(0,len(percentiles)-1)])
    print(results)