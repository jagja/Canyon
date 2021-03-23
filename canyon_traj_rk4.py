"""PART 1: Filter U,W velocity field via POD method. -> PART 2: Integrate particle paths via Runge-Kutta 4"""
from functools import partial
from multiprocessing import Pool
import time
import math
import argparse
import gc
import numpy as np

from numba import jit
import scipy.linalg as la
import scipy.interpolate

import h5py
import _pickle

@jit(nopython=True, fastmath=True)
def blin(R, U):
    """bilinear interpolation"""
    q = R-np.floor(R)
    x_z = np.array([(1-q[0])*(1-q[1]), (1-q[0])*q[1], q[0]*(1-q[1]), q[0]*q[1]])
    u_r = np.zeros(2)
    for k in range(4):
        u_r[0] += x_z[k]*U[k]
        u_r[1] += x_z[k]*U[k+4]
    return u_r

@jit(nopython=True, fastmath=True)
def total(R):
    """summation step"""
    R[0, :] += (R[1, :]+R[4, :]+2.0*(R[2, :]+R[3, :]))/6.0
    return R[0, :]

def eval_u(R, U, W, L):
    """evaluate local velocity field"""
    i, j = math.floor(R[0])%L, math.floor(R[1])
    u_w = np.array([U[i, j], U[i, j+1], U[i+1, j], U[i+1, j+1],\
                   W[i, j], W[i, j+1], W[i+1, j], W[i+1, j+1]])
    return blin(R, u_w)


def rk4(R, U, W, T, L):
    """runge kutta scheme (initial); dt = 2.0s"""
    r=np.zeros((5, 2))
    r[0, :] = R
    traj = np.zeros((T,2))
    for n in range(T):
        t = 2*n
        r[1, :] = eval_u(r[0, :], U[:, :, t], W[:, :, t], L)

        t += 1
        r[2, :] = eval_u(r[0, :]+0.5*r[1, :], U[:, :, t], W[:, :, t], L)
        r[3, :] = eval_u(r[0, :]+0.5*r[2, :], U[:, :, t], W[:, :, t], L)

        t += 1
        r[4, :] = eval_u(r[0, :]+r[3, :], U[:, :, t], W[:, :, t], L)
        r[0, :] = total(r)
         
        r[0, 0] = r[0, 0]%L

        traj[n, :] = r[0, :]

    time.sleep(0.001)
    return traj


if __name__ == '__main__':
    gc.enable()
    PARSER = argparse.ArgumentParser(description='find trajectories and exit times')

    PARSER.add_argument('-D', '--domain_height', help="height of domain (m)", \
    type=int, required=True)

    PARSER.add_argument('-d', '--diff_size', help="perturbation size (grid steps)", \
    type=int, required=True)

    PARSER.add_argument('-n', '--n_dim', help="reduced dimension", \
    type=int, required=True)

    PARSER.add_argument('-r', '--res', help="subgrid resolution", \
    type=int, required=True)

    PARSER.add_argument('-t', '--t_range', help="integration length", \
    type=int, required=True)


    PARSER.add_argument('-B', '--blocks', help="time block size", \
    type=int, required=True)

    PARSER.add_argument('-e', '--n_ens', help="ensemble number", \
    type=int, required=True)

    ARGS = PARSER.parse_args()
    H = ARGS.domain_height
    H_MAX = 32 + ARGS.diff_size
    #CHANGE THIS BACK TO H_MAX
    H_MIN = 32 - ARGS.diff_size

    DX = 1.0
    DZ = DX*H/192.0
    DT = 0.5
    T_DIFF = 5
    T_SECONDS = ARGS.t_range


    BLOCKS = ARGS.blocks
    
    N = ARGS.n_dim
    RES = ARGS.res
    TAU = math.floor((ARGS.t_range)*2/DT)
    T_0 = (T_DIFF*np.arange(ARGS.n_ens)*2/DT).astype(int)

    ASPECT = str(H)

    #read input velocity -> pod filter -> output velocity
    with open('U_'+ASPECT+'.p', 'rb') as g:
        U_OUT = _pickle.load(g, encoding='latin1')
    with open('W_'+ASPECT+'.p', 'rb') as g:
        W_OUT = _pickle.load(g, encoding='latin1')
    #rearrange U,W fields

    DOMAIN = np.abs(U_OUT[:, :, 0]) > 0.0
    CANYON = DOMAIN[:, :H_MIN]

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
    #flow=np.zeros((N_TRAJ,TAU,2)), 
    T_I = 0
    T_F = int(0.5*TAU)
    N_TRAJ = RES*RES*K_MAX
    C_SIZE = int(N_TRAJ/12)
    TEST = []
    print(N_TRAJ)

    S_0 = ord('a')

    TRAJ = np.zeros((N_TRAJ, BLOCKS*T_SECONDS, 2))


    T_START = time.time()
    for i in range(len(T_0)):

        print(str(0.5*DT*T_0[i])+' -> '+str(0.5*DT*(T_0[i]+BLOCKS*TAU))+' / '+str(0.5*DT* DIM_U[2]))

        R_0 = np.asarray(G_0)

        for j in range(BLOCKS):
            U_TAU = U_OUT[:, :, T_0[i] + j*TAU:T_0[i] + (j+1)*TAU + 1]
            W_TAU = W_OUT[:, :, T_0[i] + j*TAU:T_0[i] + (j+1)*TAU + 1]

            with Pool(processes=12) as p:
                RESULTS = p.map(partial(rk4, U=U_TAU, W=W_TAU,T=T_F-T_I, L=N_X), R_0, chunksize=C_SIZE) 
            
            R_0 = np.asarray(RESULTS)[:,-1,:]
            R_T = np.asarray(RESULTS)[:,::2,:]


            TRAJ[:, j*T_SECONDS:(j+1)*T_SECONDS, :] = R_T



        FILENAME = str('traj_' + ASPECT + '_' + str(i) + '.hdf5')

        with h5py.File(FILENAME, 'w') as f:
            f.create_dataset('trajectories_'+str(i), data=np.float32(TRAJ))
  


    T_END = time.time()
    print(T_END - T_START)
