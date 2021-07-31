"""PART 1: Filter U,W velocity field via POD method.
-> PART 2: Integrate particle paths via Runge-Kutta 4"""
import time
import math
import argparse
import gc
import numpy as np
import cupy as cp
import h5py
import _pickle


def blin(r_t, uw_t):
    """bilinear interpolation"""
    x = r_t[:, 0]-cp.floor(r_t[:, 0])
    z = r_t[:, 1]-cp.floor(r_t[:, 1])

    xz_1 = cp.multiply(1-x, 1-z)
    xz_2 = cp.multiply(1-x, z)
    xz_3 = cp.multiply(x, 1-z)
    xz_4 = cp.multiply(x, z)

    du = cp.multiply(xz_1, uw_t[0, :]) \
        + cp.multiply(xz_2, uw_t[1, :]) \
        + cp.multiply(xz_3, uw_t[2, :]) \
        + cp.multiply(xz_4, uw_t[3, :])

    dw = cp.multiply(xz_1, uw_t[4, :]) \
        + cp.multiply(xz_2, uw_t[5, :]) \
        + cp.multiply(xz_3, uw_t[6, :]) \
        + cp.multiply(xz_4, uw_t[7, :]) \

    return cp.stack([du, dw], axis=1)


def eval_u(r_t, u_t, w_t, l_x):
    """evaluate local velocity field"""

    ind_x = (cp.floor(r_t[:, 0]) % l_x).astype(cp.int_)
    ind_z = cp.floor(r_t[:, 1]).astype(cp.int_)

    uw_t = cp.array([u_t[ind_x, ind_z],
                    u_t[ind_x, ind_z+1],
                    u_t[ind_x+1, ind_z],
                    u_t[ind_x+1, ind_z+1],
                    w_t[ind_x, ind_z],
                    w_t[ind_x, ind_z+1],
                    w_t[ind_x+1, ind_z],
                    w_t[ind_x+1, ind_z+1]])

    return blin(r_t, uw_t)


def rk4_cp(r_t, u_t, w_t, tau, l_x, n_traj):
    """runge kutta scheme (initial); dt = 2.0s"""
    r_0 = r_t
    traj = cp.zeros((tau, n_traj, 2))

    for j in range(tau):
        t = 2*j
        dr_1 = eval_u(r_0, u_t[:, :, t], w_t[:, :, t], l_x)

        t += 1
        dr_2 = eval_u(r_0+0.5*dr_1, u_t[:, :, t], w_t[:, :, t], l_x)
        dr_3 = eval_u(r_0+0.5*dr_2, u_t[:, :, t], w_t[:, :, t], l_x)

        t += 1
        dr_4 = eval_u(r_0+dr_3, u_t[:, :, t], w_t[:, :, t], l_x)

        r_0 += (dr_1+dr_4+2.0*(dr_2+dr_3))/6.0

        r_0[:, 0] = r_0[:, 0] % l_x

        traj[j, :, :] = r_0

    time.sleep(1.5)
    return traj


if __name__ == '__main__':
    gc.enable()
    PARSER = argparse.ArgumentParser(description='Calculate trajectories')

    PARSER.add_argument('-D',
                        '--domain_height',
                        help="height of domain (m)",
                        type=int,
                        required=True)

    PARSER.add_argument('-d',
                        '--diff_size',
                        help="perturbation size (grid steps)",
                        type=int,
                        required=True)

    PARSER.add_argument('-r',
                        '--res',
                        help="subgrid resolution",
                        type=int,
                        required=True)

    PARSER.add_argument('-t',
                        '--t_range',
                        help="integration length",
                        type=int,
                        required=True)

    PARSER.add_argument('-e',
                        '--n_ens',
                        help="ensemble number",
                        type=int,
                        required=True)

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
    # Read input velocity -> pod filter -> output velocity
    with open('U_'+ASPECT+'.p', 'rb') as g:
        U_OUT = _pickle.load(g, encoding='latin1')
    with open('W_'+ASPECT+'.p', 'rb') as g:
        W_OUT = _pickle.load(g, encoding='latin1')

    # Rearrange U,W fields
    DOMAIN = np.abs(U_OUT[:, :, 0]) > 0.0
    CANYON = DOMAIN[:, :H_MAX]

    DIM_U = np.shape(U_OUT)
    X_ARR = DX*np.arange(DIM_U[0])
    Z_ARR = DZ*np.arange(DIM_U[1])

    N_X = len(X_ARR) - 1
    N_Z = len(Z_ARR) - 1

    # Array for finding neaRESt neighbours
    IN_CANYON = np.where(CANYON)
    G_0 = []
    K_MAX = np.sum(CANYON)
    N_TRAJ = RES*RES*K_MAX
    for k in range(K_MAX):
        for m in range(RES):
            for n in range(RES):
                G_0.append(np.array([IN_CANYON[0][k] + float(m/RES),
                                     IN_CANYON[1][k] + float(n/RES)]))

    gc.collect()

    print("Integrating trajectories")
    T_I = 0
    T_F = int(0.5*TAU)
    N_TRAJ = RES*RES*K_MAX
    print(N_TRAJ)

    TRAJ = np.zeros((T_SECONDS, N_TRAJ, 2))

    T_START = time.time()
    for i in range(len(T_0)):

        T_I = str(0.5*DT*(T_0[i]+TAU))
        T_F = str(0.5*DT*DIM_U[2])

        print(str(0.5*DT*T_0[i])+' -> '+T_I+' / '+T_F)

        R_0 = cp.asarray(G_0)

        U_TAU = cp.asarray(U_OUT[:, :, T_0[i]:T_0[i] + TAU + 1])
        W_TAU = cp.asarray(W_OUT[:, :, T_0[i]:T_0[i] + TAU + 1])

        RESULTS = rk4_cp(R_0, U_TAU, W_TAU, T_F-T_I, N_X, N_TRAJ)
        R_T = cp.asnumpy(RESULTS)[::2, :]

        del U_TAU
        del W_TAU

        TRAJ[:, :, :] = R_T
        FILENAME = str('traj_' + ASPECT + '_' + str(i) + '.hdf5')

        with h5py.File(FILENAME, 'w') as f:
            f.create_dataset('trajectories_'+str(i), data=np.float32(TRAJ))

    T_END = time.time()
    print(T_END - T_START)
