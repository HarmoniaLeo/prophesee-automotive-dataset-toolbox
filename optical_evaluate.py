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
        raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
        shape = [240,304]
        filter_boxes = filter_boxes_gen1
        classes = ['Car', "Pedestrian"]
        #percentiles = [0.0, 0.022496978317470943, 0.03584141107151823, 0.06461365563824012, 0.09765842901836184, 0.13363889435484622, 0.17335709874840652, 0.21913896857877602, 0.2795803473626733, 0.34990456407260045, 0.44142171223006244, 0.557804203925751, 0.7039533807928522, 0.8876761367290176, 1.1264612928414415, 1.4447705627987673, 1.8662489229530281, 2.4500051802140845, 3.242211733282628, 4.436053050035189, 1000]
        percentiles = [0.0, 0.09765842901836184, 0.2795803473626733, 0.7039533807928522, 1.8662489229530281, 1000]
    else:
        result_path = "/home/liubingde/100-fps-event-det/log/" + args.exp_name + "/summarise.npz"
        raw_dir = "/data/lbd/Large_Automotive_Detection_Dataset_sampling"
        shape = [720,1280]
        filter_boxes = filter_boxes_large
        classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']
        
    bbox_file = result_path
    f_bbox = np.load(bbox_file)
    dts = f_bbox["dts"]
    file_names_dt = f_bbox["file_names_dt"]
    gts = f_bbox["gts"]
    file_names_gt = f_bbox["file_names_gt"]

    # result_path = "statistics_result"
    # bbox_file = os.path.join(result_path,"gt_"+args.dataset+".npz")
    # f_bbox = np.load(bbox_file)
    # gts = f_bbox["gts"]
    # file_names_gt = f_bbox["file_names"]
    # densitys_gt = f_bbox["densitys"]

    results = []

    for i in range(0,len(percentiles)-1):
        print(i,percentiles[i],percentiles[i+1])
        dt = []
        gt = []
        for file_name in np.unique(file_names_dt):
            dt_bbox = dts[file_names_dt == file_name]
            dt_buf = []
            gt_bbox = gts[file_names_gt == file_name]
            gt_buf = []

            unique_ts, unique_indices = np.unique(gt_bbox[:,0], return_index=True)

            for bbox_count,unique_time in enumerate(unique_ts):

                gt_trans = gt_bbox[gt_bbox[:,0] == unique_time]
                dt_trans = dt_bbox[(dt_bbox[:,0] >= unique_time - args.tol) & (dt_bbox[:,0] <= unique_time + args.tol)]

                #flow = np.load(os.path.join("optical_flow_buffer",file_name + "_{0}.npy".format(int(unique_time))))

                for j in range(len(dt_trans)):
                    x, y, w, h = int(dt_trans[j,1]), int(dt_trans[j,2]), int(dt_trans[j,3]), int(dt_trans[j,4])
                    #density = np.sum(np.sqrt(flow[y:y+h,x:x+w,0]**2 + flow[y:y+h,x:x+w,1]**2))/(w * h + 1e-8)
                    #if (density >= percentiles[i]) & (density < percentiles[i+1]):
                    dt_buf.append(dt_trans[j])
                
                for j in range(len(gt_trans)):
                    x, y, w, h = int(gt_trans[j,1]), int(gt_trans[j,2]), int(gt_trans[j,3]), int(gt_trans[j,4])
                    #density = np.sum(np.sqrt(flow[y:y+h,x:x+w,0]**2 + flow[y:y+h,x:x+w,1]**2))/(w * h + 1e-8)
                    #if (density >= percentiles[i]) & (density < percentiles[i+1]):
                    gt_buf.append(gt_trans[j])
                
            dt_buf = np.vstack(dt_buf)
            gt_buf = np.vstack(gt_buf)

            dt.append(dt_buf)
            gt.append(gt_buf)
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