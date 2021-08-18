"""
https://www.ggi.infn.it/sft/SFT_2016/LectureNotes/Pollmann.pdf
"""
import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from tfi_exact import *


def itebd(Glist, llist, U, chimax):
    d = Glist[0].shape[0]
    for ibond in [0, 1]:
        ia = np.mod(ibond, 2)
        ib = np.mod(ibond + 1, 2)
        chi1 = Glist[ia].shape[1]
        chi3 = Glist[ib].shape[2]
        # Constructtheta
        theta = np.tensordot(np.diag(llist[ib]), Glist[ia], axes=(1, 1))
        theta = np.tensordot(theta, np.diag(llist[ia], 0), axes=(2, 0))
        theta = np.tensordot(theta, Glist[ib], axes=(2, 1))
        theta = np.tensordot(theta, np.diag(llist[ib], 0), axes=(3, 0))
    # ApplyU
    theta = np.tensordot(theta, np.reshape(U, (d, d, d, d)), axes=([1, 2], [0, 1]))
    # SVD
    theta = np.reshape(np.transpose(theta, (2, 0, 3, 1)), (d * chi1, d * chi3))
    X, Y, Z = np.linalg.svd(theta)
    Z = Z.T
    chi2 = np.min([np.sum(Y > 10.0 ** (-10)), chimax])
    # Truncate
    llist[ia] = Y[0:chi2] / np.sqrt(sum(Y[0:chi2] ** 2))
    X = np.reshape(X[:, 0:chi2], (d, chi1, chi2))
    Glist[ia] = np.transpose(
        np.tensordot(np.diag(llist[ib] ** (-1)), X, axes=(1, 1)), (1, 0, 2)
    )
    Z = np.transpose(np.reshape(Z[:, 0:chi2], (d, chi3, chi2)), (0, 2, 1))
    Glist[ib] = np.tensordot(Z, np.diag(llist[ib] ** (-1)), axes=(2, 0))


def bondexpectationvalue(Glist, llist, O):
    E = []
    for ibond in range(0, 2):
        ia = np.mod(ibond, 2)
        ib = np.mod(ibond + 1, 2)
        theta = np.tensordot(np.diag(llist[ib]), Glist[ia], axes=(1, 1))
        theta = np.tensordot(theta, np.diag(llist[ia], 0), axes=(2, 0))
        theta = np.tensordot(theta, Glist[ib], axes=(2, 1))
        theta = np.tensordot(theta, np.diag(llist[ib], 0), axes=(3, 0))
        thetaO = np.tensordot(
            theta, np.reshape(O, (d, d, d, d)), axes=([1, 2], [0, 1])
        ).conj()
        E.append(
            np.squeeze(
                np.tensordot(thetaO, theta, axes=([0, 1, 2, 3], [0, 3, 1, 2]))
            ).item()
        )
        return E


########Definethemodelandsimulationparameters######################
chimax = 5
delta = 0.01
N = 2000
d = 2
g = 1 
J = 1
sx = np.array([[0.0, 1.0], [1.0, 0.0]])
sz = np.array([[1.0, 0.0], [0.0, -1.0]])
H = -np.kron(sz, sz) + g * np.kron(sx, np.eye(2))
U = expm(-delta * H)

###############Initialstate:|0000>###################################
Ga = np.zeros((d, 1, 1), dtype=float)
Ga[0, 0, 0] = 1.0
Gb = np.zeros((d, 1, 1), dtype=float)
Gb[0, 0, 0] = 1.0
Glist = [Ga, Gb]
la = np.zeros(1)
la[0] = 1.0
lb = np.zeros(1)
lb[0] = 1.0
llist = [la, lb]

###############Performtheimaginarytimeevolution#######################
evals = []

for step in range(1, N):
    itebd(Glist, llist, U, chimax)
    evals.append(np.mean(bondexpectationvalue(Glist, llist, H)))

exact = infinite_gs_energy(J, g)
print(J, g, exact, evals[-1])
plt.plot(evals)
plt.axhline(exact, linestyle='--', c='black')
plt.savefig('energies.pdf')
