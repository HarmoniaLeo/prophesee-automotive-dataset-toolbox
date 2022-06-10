from itertools import count
import numpy as np
from sklearn import datasets
from src.io import npy_events_tools
from src.io import psee_loader
import os
from numpy.lib import recfunctions as rfn
from numba import jit
from joblib import Parallel, delayed

@jit(nopython=True)
def generate_leakysurface(events, q_short, p_short, q_long, p_long):
    # if memory is None:
    #     q, p = np.zeros(shape), np.zeros(shape)
    # else:
    #     q, p = memory
    t_prev = 0
    for i in range(len(events)):
        if events[i,3] == 1:
            delta = float(events[i,2] - t_prev)
            q_short = np.where(p_short - 0.0001 * delta < 0, 0, p_short - 0.0001 * delta)
            q_long = np.where(p_long - 0.000001 * delta < 0, 0, p_long - 0.000001 * delta)
            p_short = q_short
            p_long = q_long
            p_short[int(events[i,1])][int(events[i,0])] += 1
            p_long[int(events[i,1])][int(events[i,0])] += 1
        t_prev = events[i,2]
    return q_short, p_short, q_long, p_long

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = np.nonzero(dense_tensor)

    features = dense_tensor[non_zero_indices[0],non_zero_indices[1]]

    return np.stack(non_zero_indices), features

def generate_a_file(modeAndfile_name):

    mode = modeAndfile_name[0]
    file_name = modeAndfile_name[1]
    file_root = modeAndfile_name[2]

    target_dir_short = "/data2/lbd/ATIS_leaky"
    target_dir_long = "/data2/lbd/ATIS_leaky_long"

    if not os.path.exists(target_dir_short):
        os.makedirs(target_dir_short)
    if not os.path.exists(target_dir_long):
        os.makedirs(target_dir_long)

    target_root_short = os.path.join(target_dir_short, mode)
    if not os.path.exists(target_root_short):
        os.makedirs(target_root_short)
    target_root_long = os.path.join(target_dir_long, mode)
    if not os.path.exists(target_root_long):
        os.makedirs(target_root_long)


    shape = [240,304]
    target_shape = [256, 320]
    # shape = [720,1280]
    # target_shape = [512, 640]

    min_event_count = 2400000

    rh = target_shape[0] / shape[0]
    rw = target_shape[1] / shape[1]

    event_file = file_root + '_td.dat'
    bbox_file = file_root + '_bbox.npy'
    # if os.path.exists(volume_save_path):
    #     continue
    #h5 = h5py.File(volume_save_path, "w")
    f_bbox = open(bbox_file, "rb")
    start, v_type, ev_size, size, dtype = npy_events_tools.parse_header(f_bbox)
    dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
    f_bbox.close()

    unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

    f_event = psee_loader.PSEELoader(event_file)

    time_upperbound = -1e16
    count_upperbound = -1


    for bbox_count,unique_time in enumerate(unique_ts):
        print(mode,file_name, unique_time)
        volume_save_path_l_short = os.path.join(target_root_short, file_name+"_"+str(unique_time)+"_locations.npy")
        volume_save_path_f_short = os.path.join(target_root_short, file_name+"_"+str(unique_time)+"_features.npy")
        volume_save_path_l_long = os.path.join(target_root_long, file_name+"_"+str(unique_time)+"_locations.npy")
        volume_save_path_f_long = os.path.join(target_root_long, file_name+"_"+str(unique_time)+"_features.npy")
        if os.path.exists(volume_save_path_f_short) and os.path.exists(volume_save_path_l_short) and os.path.exists(volume_save_path_f_long) and os.path.exists(volume_save_path_l_long):
            continue
        end_time = int(unique_time)
        end_count = f_event.seek_time(end_time)
        if end_count is None:
            continue
        start_count = end_count - min_event_count
        if start_count < 0:
            start_count = 0
        f_event.seek_event(start_count)
        start_time = int(f_event.current_time)

        if start_time > time_upperbound:
            start_count = f_event.seek_time(start_time)
            if (start_count is None) or (start_time < 0):
                start_count = 0
        else:
            start_count = count_upperbound
            start_time = time_upperbound
            assert bbox_count > 0

        
        dat_event = f_event
        dat_event.seek_event(start_count)

        events = dat_event.load_n_events(int(end_count - start_count))
        del dat_event
        events = rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)
        events[:,0] = events[:,0] * rw
        events[:,1] = events[:,1] * rh

        if start_time > time_upperbound:
            q_short, p_short = np.zeros(target_shape, dtype = float), np.zeros(target_shape, dtype = float)
            q_long, p_long = np.zeros(target_shape, dtype = float), np.zeros(target_shape, dtype = float)

        q_short, p_short, q_long, p_long = generate_leakysurface(events, q_short, p_short, q_long, p_long)

        locations, features = denseToSparse(q_short)
        y, x = locations
        
        locations = x.astype(np.uint32) + np.left_shift(y.astype(np.uint32), 10)
        locations.tofile(volume_save_path_l_short)
        features.tofile(volume_save_path_f_short)
        
        locations, features = denseToSparse(q_long)
        y, x = locations
        
        locations = x.astype(np.uint32) + np.left_shift(y.astype(np.uint32), 10)
        locations.tofile(volume_save_path_l_long)
        features.tofile(volume_save_path_f_long)

        time_upperbound = end_time
        count_upperbound = end_count

if __name__ == '__main__':
    
        
    
    #raw_dir = "/data/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    #target_dir = "/data/lbd/ATIS_taf"
    #raw_dir = "/data/Large_Automotive_Detection_Dataset_sampling"
    #target_dir = "/data/Large_taf"
    

    total_volume_time = []
    total_taf_time = []

    raw_dir = "/data2/lbd/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"
    
    # raw_dir = "/data/Large_Automotive_Detection_Dataset_sampling"
    # target_dir = "/data/Large_taf"

    if not os.path.exists(raw_dir):
        os.makedirs(raw_dir)


    file_list = []
    
    for mode in ["train", "val", "test"]:

        file_dir = os.path.join(raw_dir, mode)
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        root = file_dir
        
        try:
            files = os.listdir(file_dir)
        except Exception:
            continue
        # Remove duplicates (.npy and .dat)
        files = [(mode, time_seq_name[:-7], os.path.join(file_dir, time_seq_name[:-7])) for time_seq_name in files
                        if time_seq_name[-3:] == 'dat']
        file_list = file_list + files

Parallel(n_jobs=-1)(delayed(generate_a_file)(file) for file in file_list)