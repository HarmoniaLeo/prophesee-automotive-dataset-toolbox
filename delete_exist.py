import os

f=open("already_exist.txt", mode='rb')
files=f.readlines()
root="/home/Large_Automotive_Detection_Dataset_processed/taf/train/feature"
for file in files:
    file = file.decode("utf-8")[:-1]
    if not os.path.exists(root+"/"+file):
        # os.remove(root+"/"+file)
        print(file)
