from src.io.psee_loader import PSEELoader
import time
import os
import random

direction = "/data/ATIS_Automotive_Detection_Dataset/detection_dataset_duration_60s_ratio_1.0/train/"
for i in range(0,100):
    lis = os.listdir(direction)
    choice = ""
    while choice[-4:] != ".dat":
        choice = random.choice(lis)
    tick = time.time()
    loader = PSEELoader(os.path.join(direction,choice))
    if time.time() - tick > 1:
        print("loader",time.time() - tick)
    tick = time.time()
    end_ts = loader.total_time()
    if time.time() - tick > 1:
        print("total_time",time.time() - tick)
    tick = time.time()
    loader.seek_event(5000)
    if time.time() - tick > 1:
        print("seek_event",time.time() - tick)
    tick = time.time()
    loader.load_n_events(50000)
    if time.time() - tick > 1:
        print("load_n_events",time.time() - tick)
    del loader