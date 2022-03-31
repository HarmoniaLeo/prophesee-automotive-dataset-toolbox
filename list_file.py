import os

files = os.listdir("/datassd4t/lbd/Large_taf/train")
files = [(file+"\n").encode("utf-8") for file in files]
f = open("already_exist.txt", mode='wb')
f.writelines(files)
f.close()
f=open("already_exist.txt", mode='rb')
files = f.readlines()
print(files[:5])