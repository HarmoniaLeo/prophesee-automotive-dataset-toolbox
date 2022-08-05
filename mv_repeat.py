import os
import shutil

f=open("already_exist.txt", mode='rb')
files = f.readlines()
files = [file.decode("utf-8")[:-1] for file in files]

old_path = "/data/lbd/Large_Automotive_Detection_Dataset_processed/taf2/train/bins4"
new_path = "/data/lbd/Large_Automotive_Detection_Dataset_processed/taf3/train/bins4"
files_exist = os.listdir(old_path)
count = 0
for file in files_exist:
    if file not in files:
        shutil.move(os.path.join(old_path,file),new_path)
        count+=1
print(count)

old_path = "/data/lbd/Large_Automotive_Detection_Dataset_processed/taf2/train/bins8"
new_path = "/data/lbd/Large_Automotive_Detection_Dataset_processed/taf3/train/bins8"
files_exist = os.listdir(old_path)
count = 0
for file in files_exist:
    if file not in files:
        shutil.move(os.path.join(old_path,file),new_path)
        count+=1
print(count)