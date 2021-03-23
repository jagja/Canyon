"""PART 1: Filter U,W velocity field via POD method. -> PART 2: Integrate particle paths via Runge-Kutta 4"""
from functools import partial
from multiprocessing import Pool
import time
import math
import argparse
import gc
import numpy as np

import scipy.linalg as la
from scipy.interpolate import interp1d

import h5py
import _pickle as pickle



if __name__ == '__main__':
    gc.enable()
    PARSER = argparse.ArgumentParser(description='find trajectories and exit times')

    PARSER.add_argument('-D', '--domain_height', help="height of domain (m)", \
    type=int, required=True)

    PARSER.add_argument('-d', '--diff_size', help="perturbation size (grid steps)", \
    type=int, required=True)


    ARGS = PARSER.parse_args()
    H = ARGS.domain_height
    H_MAX = 32 + ARGS.diff_size
    #CHANGE THIS BACK TO H_MAX
    H_MIN = 32 - ARGS.diff_size

    DX = 1.0
    DZ = DX*H/192.0
    DT = 0.5
    TS = 5


    ASPECT = str(H)

    #read input velocity -> pod filter -> output velocity
    with open('u_arr'+ASPECT+'.p', 'rb') as g:
        U_IN = pickle.load(g, encoding='latin1')
    with open('w_arr'+ASPECT+'.p', 'rb') as g:
        W_IN = pickle.load(g, encoding='latin1')
    #rearrange U,W fields

    U_IN = np.roll(np.transpose(U_IN[:, :-1, :]), 48, 0)
    W_IN = np.roll(np.transpose(W_IN[:, :-1, :]), 48, 0)
    DOMAIN = np.abs(U_IN[:, :, 0]) > 0.0
    CANYON = DOMAIN[:, :H_MIN]

    DIM_U = np.shape(U_IN)
    X_ARR = DX*np.arange(DIM_U[0])
    Z_ARR = DZ*np.arange(DIM_U[1])
   

    #interpolate velocity field in time (cubic splines)
    N_X = len(X_ARR) - 1
    N_Z = len(Z_ARR) - 1


    #principal component coefficienTS
    T_LORES = TS*np.arange(DIM_U[2])
    T_HIRES = (DT/2)*np.arange(int(np.amax(T_LORES)*2/DT))

    #VF_HIRES = np.asarray([f(T_HIRES) for f in VF])
    U_OUT = np.zeros((DIM_U[0], DIM_U[1], len(T_HIRES)))
    W_OUT = np.copy(U_OUT)
    C_X = DT/DX
    C_Z = DT/DZ

    T_START = time.time()

    for i in range(DIM_U[0]):
        for j in range(DIM_U[1]):
            F = interp1d(T_LORES, U_IN[i, j, :], kind='cubic')
            G = interp1d(T_LORES, W_IN[i, j, :], kind='cubic')
            
            U_OUT[i, j, :] = F(T_HIRES)*C_X
            W_OUT[i, j, :] = G(T_HIRES)*C_Z

        print(i)

    T_END = time.time()

    print(T_END-T_START)
   

    with open('U_'+ASPECT+'.p','wb') as f:
        pickle.dump(U_OUT,f,protocol=4)
    with open('W_'+ASPECT+'.p','wb') as g:
        pickle.dump(W_OUT,g,protocol=4)
    