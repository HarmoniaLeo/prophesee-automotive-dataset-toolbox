import os
import random

path = "/datassd4t/lbd/Large_taf/val"
files = os.listdir(path)
files = [file for file in files if "location" in file]
for file in files:
    seed = random.random()
    if seed<0.0667:
        os.remove(os.path.join(path,file[:-13]+"features.npy"))
        os.remove(os.path.join(path,file))