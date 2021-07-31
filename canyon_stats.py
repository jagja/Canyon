"""
Gather statistics like mean concentration,
winding number from trajectories
"""

import os
import numpy as np
import h5py
import _pickle

N = 152
RES = 1
H = 96
AR = 9
ASPECT = str(H)

with open('U_'+ASPECT+'.p', 'rb') as g:
    U_OUT = _pickle.load(g, encoding='latin1')

DOMAIN = np.abs(U_OUT[:, :, 0]) > 0.0
CANYON = DOMAIN[:, :32]
W_FIELD = np.zeros((AR, CANYON.shape[0], CANYON.shape[1]))
T_FIELD = np.copy(W_FIELD)


IN_CANYON = np.where(CANYON)
G_0 = []
K_MAX = np.sum(CANYON)
N_TRAJ = RES*RES*K_MAX
for k in range(K_MAX):
    for i in range(RES):
        for j in range(RES):
            G_0.append(np.array([IN_CANYON[0][k] + float(i/RES),
                       IN_CANYON[1][k] + float(j/RES)]))

R_0 = (RES * np.asarray(G_0)).astype(int)

os.chdir(ASPECT)

DATA = h5py.File('traj_'+ASPECT+'_'+str(0)+'.hdf5', 'r+')
GROUP_KEY = list(DATA.keys())[0]
TRAJ = np.asarray(list(DATA[GROUP_KEY]))


os.chdir('..')

T_RANGE = np.arange(TRAJ.shape[0])
N_STUCK = np.zeros(AR)


C_LEFT = np.zeros((AR, TRAJ.shape[0], CANYON.shape[0], CANYON.shape[1]))
C_RIGHT = np.copy(C_LEFT)

for n in range(AR):
    H = 24*(n+4)
    ASPECT = str(H)
    os.chdir(ASPECT)
    print(H)

    for i in range(N):
        DATA = h5py.File('traj_'+ASPECT+'_'+str(i)+'.hdf5', 'r+')
        GROUP_KEY = list(DATA.keys())[0]
        TRAJ = np.asarray(list(DATA[GROUP_KEY]))
        PASS = np.flip(np.where((TRAJ[-1, :, 0]-TRAJ[-2, :, 0])**2
        	                    +(TRAJ[-1, :, 1]-TRAJ[-2, :, 1])**2 > 1e-6)[0])

        N_STUCK[n] += (N_TRAJ-len(PASS)) / N

        TRAJ_X = TRAJ[:, PASS[:], 0]
        TRAJ_Z = TRAJ[:, PASS[:], 1]

        X_MEAN = np.mean(TRAJ_X, axis=0)
        Z_MEAN = np.mean(TRAJ_Z, axis=0)

        X_CENTER = TRAJ_X - np.tile(X_MEAN, (TRAJ_X.shape[0], 1))
        Z_CENTER = TRAJ_Z - np.tile(Z_MEAN, (TRAJ_Z.shape[0], 1))

        # Winding number
        DZ_SIGN = 0.5*(np.sign(Z_CENTER[1:, :])-np.sign(Z_CENTER[:-1, :]))
        WIND = np.sum(DZ_SIGN * (X_CENTER[:-1, :] > 0).astype(int), axis=0)
        T_EXIT = (np.argmax(np.abs(TRAJ_X[1:, :]-TRAJ_X[:-1, :]) > 96,
                            axis=0) - 1 + TRAJ.shape[0]) % TRAJ.shape[0]

        IN_DOMAIN = np.tile(T_RANGE,
                            (PASS.shape[0], 1)) <= T_EXIT.reshape(PASS.shape[0], 1)

        W_FIELD[n, R_0[PASS[:], 0], R_0[PASS[:], 1]] += WIND[:] / N
        T_FIELD[n, R_0[PASS[:], 0], R_0[PASS[:], 1]] += (T_EXIT[:] > 0).astype(int) / N

        X_LEFT = np.transpose(TRAJ_X < 81)
        X_RIGHT = np.transpose(TRAJ_X > 113)
        Z_BELOW = np.transpose(TRAJ_Z < 32)

        C_LEFT[n, :, R_0[PASS[:], 0], R_0[PASS[:], 1]] += (X_LEFT & Z_BELOW & IN_DOMAIN) / N
        C_RIGHT[n, :, R_0[PASS[:], 0], R_0[PASS[:], 1]] += (X_RIGHT & Z_BELOW & IN_DOMAIN) / N

    os.chdir('..')

T_LMEAN = np.sum(np.sum(T_FIELD[:, 16:81, :], axis=1),
                 axis=1) * 2 / len(PASS)

T_RMEAN = np.sum(np.sum(T_FIELD[:, 113:176, :], axis=1),
                 axis=1) * 2 / len(PASS)

W_LMEAN = np.sum(np.sum(W_FIELD[:, 16:81, :],
                 axis=1), axis=1) * 2 / len(PASS)

W_RMEAN = np.sum(np.sum(W_FIELD[:, 113:176, :], axis=1),
                 axis=1) * 2 / len(PASS)
