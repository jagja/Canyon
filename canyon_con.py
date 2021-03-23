import numpy as np
import h5py
import _pickle
import os
from matplotlib import pyplot as plt


N = 159

H = 96
ASPECT = str(H)
os.chdir(ASPECT)

DATA=h5py.File('traj_'+ASPECT+'_'+str(0)+'.hdf5','r+')
GROUP_KEY = list(DATA.keys())[0]
TRAJ = np.asarray(list(DATA[GROUP_KEY]))

C_L = np.zeros(TRAJ.shape[1])
C_R = np.copy(C_L)

MEAN_L = np.copy(C_L)
MEAN_R = np.copy(C_R) 

os.chdir('..')

for n in range(9):
    H = 24*(n+4)
    ASPECT = str(H)
    os.chdir(ASPECT)
    print(H)

    for i in range(N):
        DATA=h5py.File('traj_'+ASPECT+'_'+str(i)+'.hdf5','r+')
        GROUP_KEY = list(DATA.keys())[0]
        TRAJ = np.asarray(list(DATA[GROUP_KEY]))
        PASS = np.where((TRAJ[:,-1,0]-TRAJ[:,-2,0])**2+(TRAJ[:,-1,1]-TRAJ[:,-2,1])**2 > 1e-6)

        COUNT_L = []
        COUNT_R = []
              
        LEFT = TRAJ[PASS,:,0] < 96
        RIGHT = TRAJ[PASS,:,0] > 96
        BELOW = TRAJ[PASS,:,1] < 34   

        for k in range(TRAJ.shape[1]):
            COUNT_L.append(np.sum(np.logical_and(LEFT[:,:,k], BELOW[:,:,k])))
            COUNT_R.append(np.sum(np.logical_and(RIGHT[:,:,k], BELOW[:,:,k])))

        C_L = np.vstack((C_L, np.asarray(COUNT_L)))
        C_R = np.vstack((C_R, np.asarray(COUNT_R)))

    MEAN_L = np.vstack((MEAN_L, np.mean(C_L[1:,:], axis=0)))
    MEAN_R = np.vstack((MEAN_R, np.mean(C_R[1:,:], axis=0)))
    os.chdir('..')

MEAN_L = np.transpose(MEAN_L[1:, :])
MEAN_R = np.transpose(MEAN_R[1:, :])

FRAC_L = MEAN_L/MEAN_L[0,:]
FRAC_R = MEAN_R/MEAN_R[0,:]

