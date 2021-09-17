from scipy import integrate
from scipy.io import loadmat
from ncon import ncon
import matplotlib.pyplot as plt
from scipy.linalg import expm
import scipy as sp
from pylab import *
import numpy as np
import scipy.sparse.linalg.eigen.arpack as arp
import pylab as pl
from scipy import integrate



"""This is an attempt to implement the TDVP in simpler way than in the paper """



def init_A(D,d):

    """ Returns a random unitary """

    A,x = np.linalg.qr(np.random.rand(d*D,d*D))

    return A



def get_s(d):

    """ returns a spin up vector useful later"""

    s=np.zeros((d),dtype=np.float)

    s[0]=1.0

    return s



def get_R(A,d,D):

    """ Returns the right environment """

    s=get_s(d)

    AA=A.reshape([d,D,d,D])

    """ b) the transfer matrix """

    T = ncon([s,AA,np.conj(AA),s],([1],[1,-3,3,-1],[2,-4,3,-2],[2]))

    T = T.reshape([D*D,D*D])

    """ c) get the highest weight eigenvector """

    e,R = arp.eigs(T,k=1,which='LM')

    R = R.reshape([D,D])

    trR= np.trace(R)

    R=R/trR

    return R



def init_ising_H_bond(g,h,J):

    """ Returns bond hamiltonian"""

    sx = np.array([[0.,1.],[1.,0.]])

    sz = np.array([[1.,0.],[0.,-1.]])

    d = 2

    H = -J*np.kron(sz,sz)

    H = H + g/2.*(np.kron(sx,np.eye(d))+np.kron(np.eye(d),sx))

    H = H + g/2.*(np.kron(sx,np.eye(d))+np.kron(np.eye(d),sx))

    H = H + h/2.*(np.kron(sz,np.eye(d))+np.kron(np.eye(d),sz))

    return np.reshape(H,(d,d,d,d))



def init_U(H,d,dtau):

    """construct the time evolution operator for dt/2 and dt timesteps """

    iHdtau = -1.0j*dtau*H.reshape([d*d,d*d])*0.5

    U = expm(iHdtau)

    U2 = expm(2.0*iHdtau)

    U = U.reshape([d,d,d,d])

    U2 = U2.reshape([d,d,d,d])

    return U,U2



def get_Energy(A,d,D,H):

    """ calculate the energy of state U with Hamiltonian H """

    s=get_s(d)

    AA = A.reshape([d,D,d,D])

    R  = get_R(A,d,D)

    Energy = ncon([s,s,s,s,AA,AA,np.conj(AA),np.conj(AA),H,R],\

    ([1],[2],[3],[4],\

    [1,10,5,9],[2,11,6,10],\

    [3,12,7,9],[4,13,8,12],\

    [5,6,7,8],[11,13]))

    Energy = float(real(Energy))

    return Energy



def get_EnergyEvolved1(A,H,U,d,D):

    """ Calculate the energy after a 1st order Trotter-step """

    s=get_s(d)

    AA = A.reshape([d,D,d,D])

    R  = get_R(A,d,D)

    Top =  ncon([s,s,s,s,AA,AA,AA,AA,U,U,U],\

    ([1],[2],[5],[9],[1,3,4,-2],[2,6,7,3],[5,10,11,6],[9,-1,14,10],\

    [4,7,-3,8],[8,12,-4,-5],[11,14,12,-6]))

    Energy = ncon([Top,H,np.conj(Top),R],\

    ([7,1,2,3,5,8],[3,5,4,6],[9,1,2,4,6,8],[7,9]))

    EnergyEvolved = float(real(Energy))

    return EnergyEvolved



def get_EnergyEvolved2(A,H,U,d,D):

        """ Calculate the energy after a 1st order Trotter-step """

        s=get_s(d)

        AA = A.reshape([d,D,d,D])

        R  = get_R(A,d,D)

        Top = ncon([s,s,s,s,s,s,\

        AA,AA,AA,AA,AA,AA,\

        U,U,U,U,U],\

        ([1],[2],[5],[9],[13],[17],\

        [1,3,4,-2],[2,6,7,3],[5,10,11,6],[9,14,15,10],[13,18,19,14],[17,-1,22,18],\

        [4,7,-3,8],[8,12,-4,-5],[11,15,12,16],[16,20,-6,-7],[19,22,20,-8]))

        Energy = ncon([Top,H,np.conj(Top),R],\

        ([10,1,2,3,4,6,8,9],[4,6,5,7],[11,1,2,3,5,7,8,9],[10,11]))

        EnergyEvolved = float(real(Energy))

        return EnergyEvolved



def get_EnergyEvolved3(A,H,U,U2,d,D):

        """ Calculate the energy after a 2nd order Trotter-step """

        s=get_s(d)

        AA = A.reshape([d,D,d,D])

        R  = get_R(A,d,D)

        Top = ncon([s,s,s,s,s,s,\

        AA,AA,AA,AA,AA,AA,\

        U,U2,U,U,U2,U],\

        ([1],[2],[5],[9],[14],[19],\

        [1,3,4,-2],[2,6,7,3],[5,10,11,6],[9,15,16,10],[14,20,21,15],[19,-1,23,20],\

        [4,7,-3,8],[8,12,-4,13],[11,16,12,17],[13,18,-5,-6],[17,22,18,-7],[21,23,22,-8]))

        Energy = ncon([Top,H,np.conj(Top),R],\

        ([10,1,2,3,4,6,8,9],[4,6,5,7],[11,1,2,3,5,7,8,9],[10,11]))

        EnergyEvolved = float(real(Energy))

        return EnergyEvolved



def get_EnergyEvolved4(A,H,U,U2,d,D):

        """ Calculate the energy after a 2nd order Trotter-step """

        s=get_s(d)

        AA = A.reshape([d,D,d,D])

        R  = get_R(A,d,D)

        Top = ncon([s,s,s,s,s,s,s,s,\

        AA,AA,AA,AA,AA,AA,AA,AA,\

        U,U2,U,U,U2,U,U,U2,U],\

        ([1],[2],[5],[9],[14],[19],[24],[29],\

        [1,3,4,-2],[2,6,7,3],[5,10,11,6],[9,15,16,10],\

        [14,20,21,15],[19,25,26,20],[24,30,31,25],[29,-1,33,30],\

        [4,7,-3,8],[8,12,-4,13],[13,18,-5,-6],\

        [11,16,12,17],[17,22,18,23],[23,28,-7,-8],\

        [21,26,22,27],[27,32,28,-9],[31,33,32,-10]))

        Energy = ncon([Top,H,np.conj(Top),R],\

        ([12,1,2,3,4,5,7,9,10,11],[5,7,6,8],[13,1,2,3,4,6,8,9,10,11],[12,13]))

        EnergyEvolved = float(real(Energy))

        return EnergyEvolved



if __name__ == "__main__":

    #np.random.seed(1)

    J = 1.0

    g = 0.5

    D = 2

    d=2

    dtau=0.005



    N_T = 20



    """A=init_A(D,d)"""

    Atest= array([[-0.58372172,  0.0453531 ,  0.25329679,  0.7700992 ],

           [-0.55644694, -0.3847025 ,  0.48129677, -0.55742642],

           [-0.58892414,  0.39350079, -0.65911257, -0.25277678],

           [-0.05295386, -0.8337291 , -0.51938884,  0.17979684]])

    A=Atest

    H = init_ising_H_bond(g,0,J)

    U,U2 = init_U(H,d,dtau)

    Energy = get_Energy(A,d,D,H)

    EnergyEvolved1 = get_EnergyEvolved1(A,H,U,d,D)

    EnergyEvolved2 = get_EnergyEvolved2(A,H,U,d,D)

    EnergyEvolved3 = get_EnergyEvolved3(A,H,U,U2,d,D)

    print(Energy,EnergyEvolved1,EnergyEvolved2,EnergyEvolved3)



    En =[]

    EnEvolve1 =[]

    EnEvolve2 =[]

    EnEvolve3 =[]

    EnEvolve4 =[]

    T = []



    for i in range(1,N_T):

            U,U2 = init_U(H,d,i*dtau)

            Energy = get_Energy(A,d,D,H)

            EnergyEvolved1 = get_EnergyEvolved1(A,H,U,d,D)

            EnergyEvolved2 = get_EnergyEvolved2(A,H,U,d,D)

            EnergyEvolved3 = get_EnergyEvolved3(A,H,U,U2,d,D)

            EnergyEvolved4 = get_EnergyEvolved4(A,H,U,U2,d,D)

            En.append(np.real(Energy))

            EnEvolve1.append(np.real(EnergyEvolved1))

            EnEvolve2.append(np.real(EnergyEvolved2))

            EnEvolve3.append(np.real(EnergyEvolved3))

            EnEvolve4.append(np.real(EnergyEvolved4))

            T.append(i)

            print(i,En[-1],EnEvolve1[-1],EnEvolve2[-1],EnEvolve3[-1],EnEvolve4[-1])



    plt.plot(T,En,color='r', label='Energy')

    plt.plot(T,EnEvolve1, color='g', label='EnergyEvolved1')

    plt.plot(T,EnEvolve2, color='b', label='EnergyEvolved2')

    plt.plot(T,EnEvolve3, color='g', label='EnergyEvolved3')

    plt.plot(T,EnEvolve4, color='b', label='EnergyEvolved4')

    plt.xlabel("dt")

    plt.show()

