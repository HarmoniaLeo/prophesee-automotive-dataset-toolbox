import os
import tqdm

data_path = "/data/ATIS_h5/detection_dataset_duration_60s_ratio_1.0"

for data_folder in ['train','test']:
    final_path = os.path.join(data_path,data_folder)
    files = [item for item in os.listdir(final_path)]
    pbar = tqdm.tqdm(total=len(files), unit='File', unit_scale=True)
    for item in files:
        os.remove(os.path.join(final_path,item))