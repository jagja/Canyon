"""PART 1: Filter U,W velocity field via POD method. -> PART 2: Integrate particle paths via Runge-Kutta 4"""
from functools import partial
import time
import math
import argparse
import gc
import numpy as np
import cupy as cp

import scipy.linalg as la
import scipy.interpolate

import h5py
import _pickle


def blin(R, UW):
    """bilinear interpolation"""
    X = R[:,0]-cp.floor(R[:,0])
    Z = R[:,1]-cp.floor(R[:,1])

    XZ1 = cp.multiply(1-X,1-Z)
    XZ2 = cp.multiply(1-X,Z)
    XZ3 = cp.multiply(X,1-Z)
    XZ4 = cp.multiply(X,Z)

    DU = cp.multiply(XZ1, UW[0,:]) \
       + cp.multiply(XZ2, UW[1,:]) \
       + cp.multiply(XZ3, UW[2,:]) \
       + cp.multiply(XZ4, UW[3,:])

    DW = cp.multiply(XZ1, UW[4,:]) \
       + cp.multiply(XZ2, UW[5,:]) \
       + cp.multiply(XZ3, UW[6,:]) \
       + cp.multiply(XZ4, UW[7,:]) \

    return cp.stack([DU, DW], axis=1)


def eval_u(R, U, W, L):
    """evaluate local velocity field"""

    IND_X, IND_Z = (cp.floor(R[:,0])%L).astype(cp.int_), cp.floor(R[:,1]).astype(cp.int_)


    U_W = cp.array([U[IND_X, IND_Z], U[IND_X, IND_Z+1], U[IND_X+1, IND_Z], U[IND_X+1, IND_Z+1],\
                    W[IND_X, IND_Z], W[IND_X, IND_Z+1], W[IND_X+1, IND_Z], W[IND_X+1, IND_Z+1]])

    return blin(R, U_W)


def rk4_cp(R, U, W, T, L, N):
    """runge kutta scheme (initial); dt = 2.0s"""
    R0 = R
    # !!!! CHECK !!!!
    traj = cp.zeros((T,N,2))
    for j in range(T):
        t = 2*j
        DR1 = eval_u(R0, U[:, :, t], W[:, :, t], L)

        t += 1
        DR2 = eval_u(R0+0.5*DR1, U[:, :, t], W[:, :, t], L)
        DR3 = eval_u(R0+0.5*DR2, U[:, :, t], W[:, :, t], L)

        t += 1
        DR4 = eval_u(R0+DR3, U[:, :, t], W[:, :, t], L)
 
        R0 +=  (DR1+DR4+2.0*(DR2+DR3))/6.0

        R0[:,0] = R0[:,0] % L

        traj[j, :, :] = R0

    time.sleep(1.5)
    return traj


if __name__ == '__main__':
    gc.enable()
    PARSER = argparse.ArgumentParser(description='find trajectories and exit times')

    PARSER.add_argument('-D', '--domain_height', help="height of domain (m)", \
    type=int, required=True)

    PARSER.add_argument('-d', '--diff_size', help="perturbation size (grid steps)", \
    type=int, required=True)

    PARSER.add_argument('-r', '--res', help="subgrid resolution", \
    type=int, required=True)

    PARSER.add_argument('-t', '--t_range', help="integration length", \
    type=int, required=True)


    PARSER.add_argument('-e', '--n_ens', help="ensemble number", \
    type=int, required=True)

    ARGS = PARSER.parse_args()
    H = ARGS.domain_height
    H_MAX = 32 + ARGS.diff_size
    H_MIN = 32 - ARGS.diff_size

    DX = 1.0
    DZ = DX*H/192.0
    DT = 0.5
    T_DIFF = 5
    T_SECONDS = ARGS.t_range

    RES = ARGS.res
    TAU = math.floor((ARGS.t_range)*2/DT)
    T_0 = (T_DIFF*np.arange(ARGS.n_ens)*2/DT).astype(int)

    ASPECT = str(H)

    print("Loading (U, W) fields")
    #read input velocity -> pod filter -> output velocity
    with open('U_'+ASPECT+'.p', 'rb') as g:
        U_OUT = _pickle.load(g, encoding='latin1')
    with open('W_'+ASPECT+'.p', 'rb') as g:
        W_OUT = _pickle.load(g, encoding='latin1')
    #rearrange U,W fields

    DOMAIN = np.abs(U_OUT[:, :, 0]) > 0.0
    CANYON = DOMAIN[:, :H_MAX]

    DIM_U = np.shape(U_OUT)
    X_ARR = DX*np.arange(DIM_U[0])
    Z_ARR = DZ*np.arange(DIM_U[1])

    N_X = len(X_ARR) - 1
    N_Z = len(Z_ARR) - 1

    #array for finding neaRESt neighbours
    IN_CANYON = np.where(CANYON)
    G_0 = []
    K_MAX = np.sum(CANYON)
    N_TRAJ = RES*RES*K_MAX
    for k in range(K_MAX):
        for i in range(RES):
            for j in range(RES):
                G_0.append(np.array([IN_CANYON[0][k] + float(i/RES), \
                IN_CANYON[1][k] + float(j/RES)]))

    gc.collect()

    print("Integrating trajectories")
    T_I = 0
    T_F = int(0.5*TAU)
    N_TRAJ = RES*RES*K_MAX
    print(N_TRAJ)


    TRAJ = np.zeros((T_SECONDS, N_TRAJ, 2))


    T_START = time.time()
    for i in range(len(T_0)):

        print(str(0.5*DT*T_0[i])+' -> '+str(0.5*DT*(T_0[i]+TAU))+' / '+str(0.5*DT* DIM_U[2]))

        R_0 = cp.asarray(G_0)

        U_TAU = cp.asarray(U_OUT[:, :, T_0[i]:T_0[i] + TAU + 1])
        W_TAU = cp.asarray(W_OUT[:, :, T_0[i]:T_0[i] + TAU + 1])

        RESULTS = rk4_cp(R_0, U_TAU, W_TAU, T_F-T_I, N_X, N_TRAJ)
        R_T = cp.asnumpy(RESULTS)[::2,:]

        del U_TAU 
        del W_TAU 


        TRAJ[:, :, :] = R_T
        FILENAME = str('traj_' + ASPECT + '_' + str(i) + '.hdf5')

        with h5py.File(FILENAME, 'w') as f:
            f.create_dataset('trajectories_'+str(i), data=np.float32(TRAJ))
  


    T_END = time.time()
    print(T_END - T_START)
