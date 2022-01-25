import numpy as np
import time
from numpy.core.fromnumeric import argmin
import torch

oripoints=[
    [10,8,0],
    [2,7,1],
    [3,5,2],
    [2,6,2],
    [1,2,1],
    [4,3,0]]
afterpoints=[
    [3,1,0],
    [4,4,1],
    [6,5,2],
    [3,1,2]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

oripoints0=torch.randint(0,304,(10000,))
oripoints2=torch.randint(0,260,(10000,))
oripoints3=torch.randint(0,30,(10000,))
oripoints=torch.stack([oripoints0,oripoints2,oripoints3],dim=1)
print(oripoints.shape)
oripoints=oripoints.float()
oripoints=oripoints.to(device)

afterpoints0=torch.randint(0,304,(1024,))
afterpoints2=torch.randint(0,260,(1024,))
afterpoints3=torch.randint(0,30,(1024,))
afterpoints=torch.stack([afterpoints0,afterpoints2,afterpoints3],dim=1)
print(afterpoints.shape)
afterpoints=afterpoints.float()
afterpoints=afterpoints.to(device)

tick=time.time()
oripoints=oripoints.unsqueeze(2)
afterpoints=afterpoints.t().unsqueeze(0)
print(oripoints.shape)
print(afterpoints.shape)
dis=oripoints-afterpoints
samebatch=(dis[:,2,:]==0)
dis=torch.sqrt(dis[:,0,:]**2+dis[:,1,:]**2)
dis[~samebatch]=float('inf')
print(time.time()-tick)
dis.argmin(dim=0)