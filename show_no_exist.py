import os

f=open("already_exist.txt", mode='rb')
files = f.readlines()
files_exist = os.listdir("/data/lbd/Large_Automotive_Detection_Dataset_sampling/train")
for file in files:
    file = file.decode("utf-8")[:-1] + "_td.dat"
    if file not in files_exist:
        print(file)