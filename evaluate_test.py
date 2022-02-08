import numpy as np
from src.metrics.coco_eval import evaluate_detection
from src.io.box_loading import reformat_boxes
from src.io.box_filtering import filter_boxes

gts1=np.array([[ 666326, 287.5701 , 339.5235 , 191.57333, 151.26729, 0, 0.99402 , 0],
       [ 666326, 505.3598 , 331.8377 , 166.55472, 149.18884, 0, 0.995899, 0],
       [ 666326, 887.2574 , 327.16568, 493.80017, 213.28908, 1, 0.978683, 0],
       ])
gts2=np.array([
       [6663008, 287.19247, 339.60562, 192.10843, 151.78035, 0, 0.994329, 68],
       [6663008, 505.9741 , 331.93726, 166.4139 , 149.24467, 1, 0.996264, 63]
       ])
      #dtype=[('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')])
pred1=np.array([[ 666326, 287.5701 , 339.5235 , 191.57333, 151.26729, 0, 0.99402 , 0],
       [ 666326, 505.3598 , 331.8377 , 166.55472, 149.18884, 0, 0.995899, 0],
       [ 6663008, 887.2574 , 327.16568, 493.80017, 213.28908, 1, 0.978683, 0],
       [6663008, 505.9741 , 331.93726, 166.4139 , 149.24467, 1, 0.996264, 63]])
pred2=np.array([
       [6663008, 287.19247, 339.60562, 192.10843, 151.78035, 0, 0.994329, 68],
       [666326, 505.9741 , 331.93726, 166.4139 , 149.24467, 1, 0.996264, 63]
       ])
        #,dtype=[('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('class_confidence', '<f4'), ('track_id', '<u4')])





#RESULT_FILE_PATHS = ["file1_results_bbox.npy", "file2_results_bbox.npy"]
#GT_FILE_PATHS = ["file1_bbox.npy", "file2_bbox.npy"]

#result_boxes_list = [np.load(p) for p in RESULT_FILE_PATHS]
#gt_boxes_list = [np.load(p) for p in GT_FILE_PATHS]

# For backward-compatibility
#result_boxes_list = [reformat_boxes(p) for p in result_boxes_list]
#gt_boxes_list = [reformat_boxes(p) for p in gt_boxes_list]

gt_boxes_list=[gts1,gts2]
result_boxes_list=[pred1,pred2]

# For fair comparison with paper results
gt_boxes_list = map(filter_boxes, gt_boxes_list)
result_boxes_list = map(filter_boxes, result_boxes_list)

evaluate_detection(gt_boxes_list, result_boxes_list)