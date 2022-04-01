import os

f=open("already_exist.txt", mode='rb')
files=f.readlines()
root="/data/Large_taf/train"
for file in files:
    file = file.decode("utf-8")[:-1]
    if os.path.exists(root+"/"+file):
        os.remove(root+"/"+file)
        print(file)