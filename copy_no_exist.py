import os
import shutil

target_root = "/home/lbd/Large_Automotive_Detection_Dataset_processed/taf3"
if not os.path.exists(target_root):
    os.makedirs(target_root)
for data_division in ["train", "test", "val"]:
    if not os.path.exists(target_root + "/" + data_division):
        os.makedirs(target_root + "/" + data_division)
    for bin_division in ["bins4", "bins8"]:
        if not os.path.exists(target_root + "/" + data_division + "/" + bin_division):
            os.makedirs(target_root + "/" + data_division + "/" + bin_division)
        f=open("already_exist_" + data_division + "_" + bin_division + ".txt", mode='rb')
        existing_files=f.readlines()
        root="/home/lbd/Large_Automotive_Detection_Dataset_processed/taf2/" + data_division + "/" + bin_division
        count = 0
        for target_file in os.listdir(root):
            if target_file not in existing_files:
                # shutil.copy(root + "/" + target_file, target_root + "/" + data_division + "/" + bin_division)
                count += 1
        print(count)