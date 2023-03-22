import os

if not os.path.exists("/home/lbd/Large_Automotive_Detection_Dataset_processed/taf3"):
    os.makedirs("/home/lbd/Large_Automotive_Detection_Dataset_processed/taf3")
for data_division in ["train", "test", "val"]:
    f=open("already_exist_" + data_division + ".txt", mode='rb')
    files=f.readlines()
    root="/home/lbd/Large_Automotive_Detection_Dataset_processed/taf2/train"
    count = 0
    for file in files:
        file = file.decode("utf-8")[:-1]
        if not os.path.exists(root+"/"+file):
            # os.remove(root+"/"+file)
            print(file)
            count += 1
    print(count)