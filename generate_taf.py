import numpy as np
from sklearn import datasets
from src.io import npy_events_tools
from src.io import psee_loader
import tqdm
import os
from numpy.lib import recfunctions as rfn
import h5py

def taf_cuda(x, y, t, p, shape, volume_bins, past_volume):
    H, W = shape

    t_star = t.astype(float)[:,None]
        
    xpos = x[p == 1]
    ypos = y[p == 1]
    adderpos = np.arange(2)[None,:]
    adderpos = 1 - np.abs(adderpos-t_star[p == 1])
    adderpos = np.where(adderpos>=0,adderpos,0)

    xneg = x[p == 0]
    yneg = y[p == 0]
    adderneg = np.arange(2)[None,:]
    adderneg = 1 - np.abs(adderneg-t_star[p == 0])
    adderneg = np.where(adderneg>=0,adderneg,0)


    img_pos = np.zeros((H * W, 2)).astype(float)
    np.add.at(img_pos, W * ypos + xpos, adderpos)
    img_neg = np.zeros((H * W, 2)).astype(float)
    np.add.at(img_neg, W * yneg + xneg, adderneg)

    forward_pos = (img_pos[:,-1]==0)
    forward_neg = (img_neg[:,-1]==0)
    if not (past_volume is None):
        img_pos_old, img_neg_old, pos_ecd, neg_ecd = past_volume
        img_pos_old[:,-1] = np.where(pos_ecd[:,-1] == 0,img_pos_old[:,-1] + img_pos[:,0],img_pos_old[:,-1])
        img_neg_old[:,-1] = np.where(neg_ecd[:,-1] == 0,img_neg_old[:,-1] + img_neg[:,0],img_neg_old[:,-1])
        img_pos = np.concatenate([img_pos_old,img_pos[:,1:]],axis=1)
        pos_ecd = np.concatenate([pos_ecd,np.zeros_like(pos_ecd[:,-1:])],axis=1)
        img_neg = np.concatenate([img_neg_old,img_neg[:,1:]],axis=1)
        neg_ecd = np.concatenate([neg_ecd,np.zeros_like(neg_ecd[:,-1:])],axis=1)
        for i in range(1,img_pos.shape[1])[::-1]:
            img_pos[:,i] = np.where(forward_pos, img_pos[:,i-1],img_pos[:,i])
            pos_ecd[:,i-1] = pos_ecd[:,i-1] - 1
            pos_ecd[:,i] = np.where(forward_pos, pos_ecd[:,i-1],pos_ecd[:,i])
            img_neg[:,i] = np.where(forward_neg, img_neg[:,i-1],img_neg[:,i])
            neg_ecd[:,i-1] = neg_ecd[:,i-1] - 1
            neg_ecd[:,i] = np.where(forward_neg, neg_ecd[:,i-1],neg_ecd[:,i])
        img_pos[:,0] = np.where(forward_pos, np.zeros_like(forward_pos).astype(float), img_pos[:,0])
        pos_ecd[:,0] = np.where(forward_pos, np.zeros_like(forward_pos).astype(float) -1e6, pos_ecd[:,0])
        img_neg[:,0] = np.where(forward_neg, np.zeros_like(forward_neg).astype(float), img_neg[:,0])
        neg_ecd[:,0] = np.where(forward_neg, np.zeros_like(forward_neg).astype(float) -1e6, neg_ecd[:,0])
    else:
        pos_ecd = np.where(forward_pos, np.zeros_like(forward_pos).astype(float) -1e6, np.zeros_like(forward_pos).astype(float))
        pos_ecd = np.stack([pos_ecd,pos_ecd],axis=1)
        neg_ecd = np.where(forward_neg, np.zeros_like(forward_neg).astype(float) -1e6, np.zeros_like(forward_neg).astype(float))
        neg_ecd = np.stack([neg_ecd,neg_ecd],axis=1)
    if img_pos.shape[1] > volume_bins:
        img_pos, img_neg, pos_ecd, neg_ecd = img_pos[:,1:], img_neg[:,1:], pos_ecd[:,1:], neg_ecd[:,1:]

    histogram = np.concatenate([img_neg, img_pos], -1).reshape((H, W, img_neg.shape[1] * 2)).transpose(2, 0, 1)
    ecd = np.concatenate([neg_ecd, pos_ecd], -1).reshape((H, W, img_neg.shape[1] * 2)).transpose(2, 0, 1)
    past_volume = (img_pos, img_neg, pos_ecd, neg_ecd)
    return histogram, ecd, past_volume

def generate_taf_cuda(events, shape, past_volume = None, volume_bins=5):
    x, y, t, p, z = events.T

    x, y, p = x.astype(int), y.astype(int), p.astype(int)
    
    if past_volume is None:
        for bin in range(volume_bins):
            x_, y_, t_, p_ = x[z == bin], y[z == bin], t[z == bin], p[z == bin]
            histogram, ecd, past_volume = taf_cuda(x_, y_, t_, p_, shape, volume_bins, past_volume)
    else:
        histogram, ecd, past_volume = taf_cuda(x, y, t, p, shape, volume_bins, past_volume)

    return np.stack([histogram, ecd], -1), past_volume

def denseToSparse(dense_tensor):
    """
    Converts a dense tensor to a sparse vector.

    :param dense_tensor: BatchSize x SpatialDimension_1 x SpatialDimension_2 x ... x FeatureDimension
    :return locations: NumberOfActive x (SumSpatialDimensions + 1). The + 1 includes the batch index
    :return features: NumberOfActive x FeatureDimension
    """
    non_zero_indices = np.nonzero(np.abs(dense_tensor).sum(axis=-1))

    print(non_zero_indices)
    select_indices = non_zero_indices.split(1, axis=1)
    features = np.squeeze(dense_tensor[select_indices], axis=-2)

    return non_zero_indices, features

min_event_count = 200000
events_window = 50000
events_window_abin = 10000
event_volume_bins = 5
shape = [240,304]
raw_dir = "/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0"

for mode in ["train","val","test"]:
    
    file_dir = os.path.join(raw_dir, mode)
    root = file_dir
    target_root = os.path.join("/data/ATIS_taf", mode)
    #h5 = h5py.File(raw_dir + '/ATIS_taf_'+mode+'.h5', 'w')
    files = os.listdir(file_dir)
    # Remove duplicates (.npy and .dat)
    files = [time_seq_name[:-7] for time_seq_name in files
                    if time_seq_name[-3:] == 'dat']

    pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)

    sequence_start_t = []
    sequence_start_n = []
    sequence_end_t = []
    sequence_end_n = []
    file_names = []

    route = os.path.join(root, 'buffer_t{0}_n{1}.npz'.format(events_window,min_event_count))

    for i_file, file_name in enumerate(files):
        event_file = os.path.join(root, file_name + '_td.dat')
        bbox_file = os.path.join(root, file_name + '_bbox.npy')
        f_bbox = open(bbox_file, "rb")
        start, v_type, ev_size, size = npy_events_tools.parse_header(f_bbox)
        dat_bbox = np.fromfile(f_bbox, dtype=v_type, count=-1)
        f_bbox.close()

        unique_ts, unique_indices = np.unique(dat_bbox['t'], return_index=True)

        f_event = psee_loader.PSEELoader(event_file)

        time_upperbound = -1e16
        already = False
        for bbox_count,unique_time in enumerate(unique_ts):
            if unique_time <= 500000:
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
            if (end_time - start_time) < events_window:
                start_time = end_time - events_window
            else:
                start_time = end_time - (int((end_time - start_time - events_window)/events_window_abin) + 1) * events_window_abin - events_window

            if start_time > time_upperbound:
                start_count = f_event.seek_time(start_time)
                if (start_count is None) or (start_time < 0):
                    start_count = 0
                sequence_start_t.append(start_time)
                sequence_start_n.append(start_count)
                file_names.append(file_name)
                #print("start_append")
                if already:
                    sequence_end_n.append(count_upperbound)
                    sequence_end_t.append(time_upperbound)
                already = True

            dat_event = f_event
            dat_event.seek_event(start_count)
            events = dat_event.load_n_events(int(end_count - start_count))
            del dat_event
            events = rfn.structured_to_unstructured(events)[:, [1, 2, 0, 3]].astype(float)

            z = np.zeros_like(events[:,0])

            bins = int((end_time - start_time) / events_window_abin)
            assert bins == (end_time - start_time) / events_window_abin
            
            for i in range(bins):
                z = np.where((events[:,2] >= start_time + i * events_window_abin)&(events[:,2] <= start_time + (i + 1) * events_window_abin), i, z)
                #events_timestamps.append(start_time + (i + 1) * self.events_window_abin)
            events = np.concatenate([events,z[:,None]], axis=1)

            if start_time > time_upperbound:
                memory = None
                events_ = events[events[...,4] < event_volume_bins]
                volume, memory = generate_taf_cuda(events_, shape, memory, event_volume_bins)
                iter = event_volume_bins
            else:
                iter = 0
            while(iter < bins):
                events_ = events[events[...,4] == iter]
                volume, memory = generate_taf_cuda(events_, shape, memory, event_volume_bins)
                iter += 1
            #h5.create_dataset(file_name+"/"+str(unique_time), shape = volume.shape, data = volume)
            volume_ = volume.copy()
            volume_[...,1] = np.where(volume_[...,1]>-1e6, volume_[...,1] - 1, 0)
            volume_ = volume_.transpose(1,2,3,0)
            locations, features = denseToSparse(volume_)
            volume_save_path = os.path.join(target_root, file_name+"_"+str(unique_time)+".npz")
            np.save(volume_save_path, locations = locations, features = features)

            time_upperbound = end_time
            count_upperbound = end_count

        pbar.update(1)

        if count_upperbound > 0:
            sequence_end_n.append(count_upperbound)
            sequence_end_t.append(time_upperbound)
        #print("end_append_last")


        assert len(sequence_end_n) == len(sequence_end_t) == len(sequence_start_n) == len(sequence_start_t) == len(file_names)
    
    np.savez(route,sequence_start_t = np.array(sequence_start_t),
                sequence_end_t = np.array(sequence_end_t),
                sequence_start_n = np.array(sequence_start_n),
                sequence_end_n = np.array(sequence_end_n),
                file_name = np.array(file_names))
    print("Save buffer to "+route)
    pbar.close()
    #h5.close()