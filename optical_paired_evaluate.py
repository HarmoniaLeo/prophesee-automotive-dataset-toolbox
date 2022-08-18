import numpy as np
from sympy import EX
from torch import exp_
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import argparse
from src.io.box_filtering import filter_boxes_gen1, filter_boxes_large
from src.metrics.coco_eval import evaluate_detection
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='visualize one or several event files along with their boxes')
    parser.add_argument('-dataset', type=str)
    parser.add_argument('-tol', type = int, default=4999)

    args = parser.parse_args()
    mode = "test"

    results_slow = {"File Name": [], "Time stamp": []}
    results_fast = {"File Name": [], "Time stamp": []}

    if args.dataset == "gen1":
        exp_names = ["basic_long_pp","basic_long500000_pp","basic_long1000000_pp","basic_frame50000","basic_frame100000","basic_frame200000",
        "basic_leaky1e5","basic_leaky25e6","basic_leaky1e6","taf_bins4","taf_bins8","taf_bins4_tcn_connect","taf_bins8_tcn_connect"]
        result_paths = []
        for exp_name in exp_names:
            result_paths.append("/home/lbd/100-fps-event-det/log/" + exp_name + "/summarise_stats.npz")
        shape = [240,304]
        filter_boxes = filter_boxes_gen1
        classes = ['Car', "Pedestrian"]
        #percentiles = [0.0, 0.023955019742690712, 0.042253373088964444, 0.07143436976739469, 0.101650021925867, 0.13346816647302767, 0.16817225179138567, 0.20777102878868156, 0.2576741573299752, 0.317873322207675, 0.39235074076386517, 0.4885132853135831, 0.613289039892126, 0.7680029836511075, 0.9726207443471735, 1.2624602141157863, 1.6633663775961032, 2.234325892848631, 3.051188316051193, 4.3239727771346255]
        #percentiles = [0.0, 0.022090551140021526, 0.03307005296535371, 0.05671614855808474, 0.08410216123034962, 0.11406982861849572, 0.14831617911666944, 0.18462005594938136, 0.22870882848611626, 0.2853853858727559, 0.3554077913283515, 0.44819659632660175, 0.5715129938105777, 0.7367699321420207, 0.9613864329222702, 1.2917300269541516, 1.7416549267996466, 2.4035640687750264, 3.302250419545379, 4.655760829519853, 1000]
        #percentiles = [0.0, 0.03307005296535371, 0.08410216123034962, 0.14831617911666944, 0.22870882848611626, 0.3554077913283515, 0.5715129938105777, 0.9613864329222702, 1.7416549267996466, 3.302250419545379, 1000]
        #[0.0, 0.022826836845540655, 0.037013073408681124, 0.06448076630071245, 0.09472751189131885, 0.12681692210188172, 0.16266472495121756, 0.20304601001680278, 0.2538587115258659, 0.3154112080140129, 0.39181042411402056, 0.4893193445564121, 0.6169536673563197, 0.7805525190114224, 0.9951987812240594, 1.2969357872594736, 1.703355726917305, 2.296600946458227, 3.121383262000021, 4.374163669844856]
        #percentiles = [0.0, 0.08410216123034962, 0.22870882848611626, 0.5715129938105777, 1.7416549267996466, 1000]
        #percentiles = [0.0, 0.101650021925867, 1.6633663775961032, 1000]
        percentiles2 = [0.0, 0.12681692210188172, 1.2969357872594736, 1000]
        percentiles1 = [0.0, 0.09472751189131885, 0.2538587115258659, 0.6169536673563197, 1.703355726917305, 1000]
    else:
        exp_names = ["gen4_basic_long_pp","gen4_basic_long500000_pp","gen4_basic_long1000000_pp","gen4_basic_frame400000","gen4_basic_frame800000",
        "gen4_basic_frame1000000","gen4-taf2","gen4-taf_tcn_connect2","gen4_taf_bins8","gen4_taf_tcn_connect_bins8"]
        result_paths = []
        for exp_name in exp_names:
            result_paths.append("/home/liubingde/100-fps-event-det/log/" + exp_name + "/summarise_stats.npz")
        shape = [720,1280]
        filter_boxes = filter_boxes_large
        classes = ['pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light']
        #percentiles = [0.0, 0.13798373765648486, 0.6976278461290158, 2.025066711356914, 5.278082102864997, 1000]
        #percentiles = [0.0, 0.0, 0.0, 0.017224068593383286, 0.04395048958884435, 0.10717341445938622, 0.20183185315356414, 0.31609882595677286, 0.4509914337597868, 0.6113675070465179, 0.811929173028783, 1.0687768126032453, 1.4313008685647621, 1.9108096336945133, 2.5080834649774104, 3.238813250464127, 4.108326745357363, 5.203311921389142, 6.6210589062068905, 8.648134991471418]
        #percentiles = [0.0, 0.0, 0.0, 0.018915752892064257, 0.0489449954503074, 0.11323120410190246, 0.20295585924830123, 0.3099795391225571, 0.4390496015194756, 0.59493510328393, 0.78934969219668, 1.0402726076857434, 1.3904864304128026, 1.862398078728036, 2.463074497303443, 3.211974451696608, 4.130296275763464, 5.292283327513696, 6.754746824399112, 8.873209713168537]
        percentiles2 = [0.0, 0.13691002283099496, 3.2786494067846, 1000]
        #percentiles2 = [0.0, 0.0, 0.0, 0.024069758977451696, 0.061864120261698595, 0.13691002283099496, 0.23318884073192203, 0.34242255943500355, 0.47486729209948575, 0.6336573219372644, 0.8326946149540354, 1.0854665004882589, 1.4415784200310098, 1.9203229983640102, 2.529642175121489, 3.2786494067846, 4.20493449274388, 5.370377098119524, 6.799594222218903, 8.962206297968446]
        percentiles1 = [0.0, 0.061864120261698595, 0.47486729209948575, 1.4415784200310098, 4.20493449274388, 1000]
    
    dts_list = []
    file_names_dt_list = []
    densitys_dt_list = []
    for i,exp_name in enumerate(exp_names):
        results_slow[exp_name] = []
        results_fast[exp_name] = []
        bbox_file = result_paths[i]
        f_bbox = np.load(bbox_file)
        dts_list.append(f_bbox["dts"])
        file_names_dt_list.append(f_bbox["file_names"])
        densitys_dt_list.append(f_bbox["densitys"])
    
    result_path = "statistics_result"
    bbox_file = os.path.join(result_path,"gt_"+args.dataset+".npz")
    f_bbox = np.load(bbox_file)
    gts = f_bbox["gts"]
    file_names_gt = f_bbox["file_names"]
    densitys_gt = f_bbox["densitys"]

    for i in [0, 3]:
        print(i,percentiles1[i],percentiles1[i+1])

        for i_file, file_name in enumerate(np.unique(file_names_gt)):

            gt_bbox = gts[(file_names_gt == file_name)&(densitys_gt >= percentiles1[i])&(densitys_gt < percentiles1[i+1])]
            if len(gt_bbox) == 0:
                continue

            for unique_ts in np.unique(gt_bbox[:,0]):
                gt_bbox_t = gt_bbox[gt_bbox[:,0] == unique_ts]
                for j, dts in enumerate(dts_list):
                    dt_bbox = dts[(file_names_dt_list[j] == file_name)&(densitys_dt_list[j] >= percentiles1[i])&(densitys_dt_list[j] < percentiles1[i+1])]
                    dt_bbox_t = dt_bbox[dt_bbox[:,0] == unique_ts]
                    dt = [dt_bbox_t]
                    gt = [gt_bbox_t]
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
                    if len(gt_boxes_list1) == 0:
                        continue
                    result = evaluate_detection(gt_boxes_list1, result_boxes_list1, time_tol = args.tol, classes=classes,height=shape[0],width=shape[1])
                    if i == 0:
                        results_slow[exp_names[j]].append(result)
                    else:
                        results_fast[exp_names[j]].append(result)
                lens = []
                for ls in results_slow.values():
                    lens.append(len(ls))
                if lens[1:] != lens[:-1]:
                    if i == 0:
                        results_slow["File Name"].append(file_name)
                        results_slow["Time stamp"].append(unique_ts)
                    else:
                        results_fast["File Name"].append(file_name)
                        results_fast["Time stamp"].append(unique_ts)
                lens = []
                for ls in results_slow.values():
                    lens.append(len(ls))
                if lens[1:] != lens[:-1]:
                    print(lens)
                    raise Exception("Break")
                
    results_slow = pd.DataFrame(results_slow).to_csv("Result_slow.csv")
    results_fast = pd.DataFrame(results_fast).to_csv("Result_fast.csv")