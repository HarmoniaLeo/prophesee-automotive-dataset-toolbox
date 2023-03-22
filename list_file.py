import os

for data_division in ["train", "test", "val"]:
    files = os.listdir("/data/lbd/Large_Automotive_Detection_Dataset_sampling/" + data_division)
    files = [(file[:-9]+"\n").encode("utf-8") for file in files]
    f = open("already_exist_" + data_division +".txt", mode='wb')
    f.writelines(files)
    f.close()
    f=open("already_exist_" + data_division +".txt", mode='rb')
    files = f.readlines()
    print(files[:5])