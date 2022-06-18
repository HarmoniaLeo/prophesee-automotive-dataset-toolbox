import os

f=open("already_exist.txt", mode='rb')
files=f.readlines()
root="/home/lbd/Large_Automotive_Detection_Dataset_processed/taf/train/feature"
count = 0
for file in files:
    file = file.decode("utf-8")[:-1]
    if not os.path.exists(root+"/"+file):
        # os.remove(root+"/"+file)
        print(file)
        count += 1
print(count)