"""
---------------------------------------------------------------------
Script file for initializing the Hamiltonian and MPS tensors before passing to
the TEBD routine.

by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 06/2020
"""

# preamble
import numpy as np
from doTEBD import doTEBD
from xmps.ncon import ncon
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

""" Example 1: XX model """

# set bond dimensions and simulation options
chi = 2  # bond dimension
tau = 0.01  # timestep

numiter = 500  # number of timesteps
evotype = "real"  # real or imaginary time evolution
E0 = -4 / np.pi  # specify exact ground energy (if known)
midsteps = 1#int(1 / tau)  # timesteps between MPS re-orthogonalization

# define Hamiltonian (quantum XX model)
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0, -1]])
# hamAB = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
# hamBA = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
g = 1
J = 1
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
hamAB = (-np.kron(sZ, sZ) + g * np.kron(sX, np.eye(2))).reshape(2, 2, 2, 2)
hamBA = (-np.kron(sZ, sZ) + g * np.kron(sX, np.eye(2))).reshape(2, 2, 2, 2)

# initialize tensors
d = hamAB.shape[0]
sAB = np.ones(chi) / np.sqrt(chi)
sBA = np.ones(chi) / np.sqrt(chi)
A = np.random.rand(chi, d, chi)
B = np.random.rand(chi, d, chi)

""" Real time evolution with TEBD """
# run TEBD routine
A, B, sAB, sBA, rhoAB, rhoBA, all_evs = doTEBD(
    hamAB,
    hamBA,
    A,
    B,
    sAB,
    sBA,
    chi,
    tau,
    evotype=evotype,
    numiter=numiter,
    midsteps=midsteps,
    E0=E0,
    measure_ops=[
        np.kron(np.eye(2), sX),
        np.kron(np.eye(2), sY),
        np.kron(np.eye(2), sZ),
    ],
)

plt.plot(all_evs)
plt.savefig('time_evo.pdf')
