import h5py
import numpy as np
import os
from src.io import npy_events_tools, dat_events_tools
from numpy.lib import recfunctions as rfn
from numpy import dtype

# with h5py.File("/Users/harmonialeo/Downloads/0000-2/0000.h5",'r') as f:
#     print(np.unique(f["events"][:73000,0]))

for i in range(0,1):
    file = "/home/lbd/v2e/output/" + str(i).zfill(4) + "/" + str(i).zfill(4) + ".h5"
    if i in [0, 3, 6, 7, 17, 18]:
        target = open("/data2/lbd/kitti/train" + str(i).zfill(4) + "_td.dat", "wb")
    else:
        target = open("/data2/lbd/kitti/test" + str(i).zfill(4) + "_td.dat", "wb")
    with h5py.File(file,'r') as f:
        events = f["events"]
        events = rfn.unstructured_to_structured(events, dtype = dtype([('t', '<u8'), ('x', '<f4'), ('y', '<f4'), ('p', '<i4')]))
        dat_events_tools.write_event_buffer(target, events)
    target.close()