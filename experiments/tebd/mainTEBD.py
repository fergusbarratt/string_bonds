"""
mainTEBD.py
---------------------------------------------------------------------
Script file for initializing the Hamiltonian and MPS tensors before passing to
the TEBD routine.

by Glen Evenbly (c) for www.tensors.net, (v1.2) - last modified 06/2020
"""

# preamble
import numpy as np
from doTEBD import doTEBD
from xmps.ncon import ncon

""" Example 1: XX model """

# set bond dimensions and simulation options
chi = 16  # bond dimension
tau = 0.1  # timestep

numiter = 500  # number of timesteps
evotype = "imag"  # real or imaginary time evolution
E0 = -4 / np.pi  # specify exact ground energy (if known)
midsteps = int(1 / tau)  # timesteps between MPS re-orthogonalization

# define Hamiltonian (quantum XX model)
sX = np.array([[0, 1], [1, 0]])
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0, -1]])
#hamAB = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
#hamBA = (np.real(np.kron(sX, sX) + np.kron(sY, sY))).reshape(2, 2, 2, 2)
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

""" Imaginary time evolution with TEBD """
# run TEBD routine
A, B, sAB, sBA, rhoAB, rhoBA = doTEBD(hamAB, hamBA, A, B, sAB, sBA, chi,
    tau, evotype=evotype, numiter=numiter, midsteps=midsteps, E0=E0)

# continute running TEBD routine with reduced timestep
tau = 0.01
numiter = 2000
midsteps = 100
A, B, sAB, sBA, rhoAB, rhoBA = doTEBD(hamAB, hamBA, A, B, sAB, sBA, chi,
    tau, evotype=evotype, numiter=numiter, midsteps=midsteps, E0=E0)

# continute running TEBD routine with reduced timestep and increased bond dim
chi = 32
tau = 0.001
numiter = 20000
midsteps = 1000
A, B, sAB, sBA, rhoAB, rhoBA = doTEBD(hamAB, hamBA, A, B, sAB, sBA, chi,
    tau, evotype=evotype, numiter=numiter, midsteps=midsteps, E0=E0)

# compare with exact results
energyMPS = np.real(0.5 * ncon([hamAB, rhoAB], [[1, 2, 3, 4], [1, 2, 3, 4]]) +
                    0.5 * ncon([hamBA, rhoBA], [[1, 2, 3, 4], [1, 2, 3, 4]]))
enErr = abs(energyMPS - E0)
print('Final results => Bond dim: %d, Energy: %f, Energy Error: %e' %
      (chi, energyMPS, enErr))
