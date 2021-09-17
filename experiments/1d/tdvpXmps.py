from xmps.iMPS import iMPS
from qmps.ground_state import Hamiltonian
from scipy.linalg import expm
import numpy as np
from ansatze import U_full
from qmps.tools import unitary_to_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt

def tdvp_update(mps:iMPS, W, ϵ=1e-2):
    B = mps.dA_dt([W])
    mps += ϵ*B 
    mps.left_canonicalise()
    return mps


def time_evolve(initMPS:iMPS, numSteps, trackSingleOps, hamiltonian, timeEvo):
    """
    trackSingleOps: list of array of single site operators to track
    trackDoubleOps: list of array of two site operators to track

    """
    mps = initMPS

    singleResults = []
    energy = []

    for _ in tqdm(range(numSteps)):
        
        mps = tdvp_update(mps, timeEvo)
        singleResults.append(mps.Es(trackSingleOps))
        energy.append(mps.energy([hamiltonian]))

    return np.array(singleResults), energy


def main():
    random_seed = 500
    np.random.seed(random_seed)
    initParams = np.random.rand(15)
    A = iMPS([unitary_to_tensor(U_full(initParams))])

    H = Hamiltonian({"ZZ":-1, "X":0.5}).to_matrix()
    
    totalTime = 2
    numSteps = 1000
    allSteps = np.linspace(0,totalTime, numSteps)
    dt = allSteps[1] - allSteps[0]

    W = expm(-1j * H * 1e-1)

    X = np.array([
        [0,1],
        [1,0]
    ])

    Y = np.array([
        [0,-1j],
        [1j,0]
    ])

    Z = np.array([
        [1,0],
        [0,-1]
    ])
    
    singleOps = [X,Y,Z]

    singleResults, energy = time_evolve(
        A, numSteps, singleOps, H, W
    )

    Xs = singleResults[:,0]
    Ys = singleResults[:,1]
    Zs = singleResults[:,2]

    fig, (ax1, ax2) = plt.subplots(2,1)

    ax1.plot(energy)
    ax1.title.set_text('Energy Drift')

    ax2.plot(Xs, "r", label = "⟨X⟩")
    ax2.plot(Ys, "y", label = "⟨Y⟩")
    ax2.plot(Zs, "b", label = "⟨Z⟩")
    
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
            
