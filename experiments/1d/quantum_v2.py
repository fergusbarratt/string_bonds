import cirq
from qmps.tools import (
    environment_to_unitary,
    unitary_to_tensor
)

from scipy.linalg import cholesky, expm
from xmps.iMPS import TransferMatrix
from xmps.Hamiltonians import Hamiltonian
from ansatze import U_full
import numpy as np
from scipy.optimize import minimize
from functools import partial
import sys
import matplotlib.pyplot as plt
import tqdm


class Full_U_Gate(cirq.Gate):
    def __init__(self, p):
        self.p = p
        self.U = U_full(p)

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ["A","A"]

    def _unitary_(self):
        return self.U
 

class Full_V_Gate(cirq.Gate):
    def __init__(self, v):
        self.v = v
        self.n_qbs = int(np.log2(v.shape[0])) 

    def num_qubits(self):
        return self.n_qbs

    def _circuit_diagram_info_(self, args):
        return ["V"] * self.n_qbs

    def _unitary_(self):
        return self.v


class Ham(cirq.Gate):
    def __init__(self, H):
        self.h = H

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ["H","H"]

    def _unitary_(self):
        return self.h


class DoubleHam(cirq.Gate):
    def __init__(self, H):
        self.h = H

    def num_qubits(self):
        return 2

    def _circuit_diagram_info_(self, args):
        return ["HH","HH"]

    def _unitary_(self):
        return self.h


def get_env_exact(U):
    """get_env_exact: v. useful for testing. Much faster than variational optimization of the env.

    :param U:
    """
    _, _, r = TransferMatrix(unitary_to_tensor(U)).eigs()
    return environment_to_unitary(cholesky(r).conj().T)


def build_no_time_evo_rdm(U, V):
    """
    Build this circuit:

-0 |0⟩--------|---------------
-             U
-1 |0⟩----|---|---------------  σ0
-         U       
-2 |0⟩-|--|-------------------  σ1
-      V              
-3 |0⟩-|----------------------  
    
    """
 
    qubits = cirq.GridQubit.rect(1,4)
    m1 = cirq.Moment([V.on(*qubits[2:])])
    m2 = cirq.Moment([U.on(*qubits[1:3])])
    m3 = cirq.Moment([U.on(*qubits[0:2])])

    return cirq.Circuit((m1,m2,m3))


def build_time_evo_rdm(U,V,H):
    """
    Build this circuit:
-0 |0⟩----------------|-------
-                     U
-1 |0⟩--------------|-|-|-----
-                   U   H
-2 |0⟩----------|---|---|---|-  σ0
-               U           H
-3 |0⟩-----|----|-------|---|-  σ1
-          U            H
-4 |0⟩-|---|------------|-----  
-      V
-5 |0⟩-|----------------------  
    """

    qubits = cirq.GridQubit.rect(1,6)
    m1 = cirq.Moment([V.on(*qubits[4:])])
    m2 = cirq.Moment([U.on(*qubits[3:5])])
    m3 = cirq.Moment([U.on(*qubits[2:4])])
    m4 = cirq.Moment([U.on(*qubits[1:3])])
    m5 = cirq.Moment([U.on(*qubits[0:2])])
    m6 = cirq.Moment([H.on(*qubits[1:3]), H.on(*qubits[3:5])])
    m7 = cirq.Moment([H.on(*qubits[2:4])])

    return cirq.Circuit((m1,m2,m3,m4,m5,m6,m7))
    

def build_second_order_time_evo_rdm(U,V,H,H2):
    """Build this circuit:
-0 |0⟩-------------|-----------
-                  U
-1 |0⟩-----------|-|-|---------
-                U   H
-2 |0⟩---------|-|---|--||-----
-              U        H
-3 |0⟩-------|-|-----|--||--|--  σ0
-            U       H      H
-4 |0⟩-----|-|-------|--||--|--  σ1
-          U            H
-5 |0⟩---|-|---------|--||-----  
-        U           H
-6 |0⟩-|-|-----------|--------- 
-      V
-7 |0⟩-|-----------------------"""

    qubits = cirq.GridQubit.rect(1,8)

    m1 = cirq.Moment([V.on(*qubits[6:])])
    m2 = cirq.Moment([U.on(*qubits[5:7])])
    m3 = cirq.Moment([U.on(*qubits[4:6])])
    m4 = cirq.Moment([U.on(*qubits[3:5])])
    m5 = cirq.Moment([U.on(*qubits[2:4])])
    m6 = cirq.Moment([U.on(*qubits[1:3])])
    m7 = cirq.Moment([U.on(*qubits[0:2])])
    
    m8 = cirq.Moment([
        H.on(*qubits[1:3]),
        H.on(*qubits[3:5]),
        H.on(*qubits[5:7]),
        ])

    m9 = cirq.Moment([
        H2.on(*qubits[2:4]),
        H2.on(*qubits[4:6]),
        ])

    m10 = cirq.Moment([H.on(*qubits[3:5])])

    return cirq.Circuit((m1,m2,m3,m4,m5,m6,m7,m8,m9,m10))


def build_second_order_time_evo_rdm_alternative(U,V,H,H2):
    """Build this circuit:
-0 |0⟩-----------------|--------
-                      U
-1 |0⟩---------------|-|-|------
-                    U   H
-2 |0⟩-------------|-|---|-||---
-                  U       H
-3 |0⟩-----------|-|-----|-||-|-
-                U       H    H
-4 |0⟩---------|-|-------|-||-|- σ0
-              U           H
-5 |0⟩-------|-|---------|-||-|- σ1
-            U           H    H
-6 |0⟩-----|-|-----------|-||-|-  
-          U               H
-7 |0⟩---|-|-------------|-||---  
-        U               H
-8 |0⟩-|-|---------------|------ 
-      V
-9 |0⟩-|------------------------"""
    qubits = cirq.GridQubit.rect(1,10)

    m1 = cirq.Moment([V.on(*qubits[8:])])
    m2 = cirq.Moment([U.on(*qubits[7:9])])
    m3 = cirq.Moment([U.on(*qubits[6:8])])
    m4 = cirq.Moment([U.on(*qubits[5:7])])
    m5 = cirq.Moment([U.on(*qubits[4:6])])
    m6 = cirq.Moment([U.on(*qubits[3:5])])
    m7 = cirq.Moment([U.on(*qubits[2:4])])
    m8 = cirq.Moment([U.on(*qubits[1:3])])
    m9 = cirq.Moment([U.on(*qubits[0:2])])
    
    m10 = cirq.Moment([
        H.on(*qubits[1:3]),
        H.on(*qubits[3:5]),
        H.on(*qubits[5:7]),
        H.on(*qubits[7:9]),
        ])

    m11 = cirq.Moment([
        H2.on(*qubits[2:4]),
        H2.on(*qubits[4:6]),
        H2.on(*qubits[6:8]),
        ])

    m12 = cirq.Moment([
        H.on(*qubits[3:5]),
        H.on(*qubits[5:7])

        ])

    return cirq.Circuit((m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12))



def energy_from_rdm_params(p, H):

    V = get_env_exact(U_full(p))
    c1 = build_no_time_evo_rdm(Full_U_Gate(p), Full_V_Gate(V)) 
    qubits = cirq.GridQubit.rect(1,5)
    sim = cirq.Simulator(dtype = np.complex128)

    rho = sim.simulate(c1).density_matrix_of(qubits[1:3])

    return np.trace(rho@H).real


def compare_energy_change_trotter(dt,H, params):

    U = U_full(params)
    V = get_env_exact(U)

    W = expm(-1.0j * H * dt)
    W1= expm(-1.0j * H * dt/2)
    
    c1 = build_no_time_evo_rdm(
        Full_U_Gate(params),
        Full_V_Gate(V)
        )

    c2 = build_time_evo_rdm(
        Full_U_Gate(params), 
        Full_V_Gate(V), 
        Ham(W))

    c3 = build_second_order_time_evo_rdm(
        Full_U_Gate(params), 
        Full_V_Gate(V), 
        Ham(W1),
        DoubleHam(W)
    )
    

    qubits = cirq.GridQubit.rect(1,5)
    sim = cirq.Simulator(dtype=np.complex128)

    rho = sim.simulate(c1).density_matrix_of(qubits[1:3])
    sigma = sim.simulate(c2).density_matrix_of(qubits[2:4])
    tao = sim.simulate(c3).density_matrix_of(qubits[3:5])
    

    E1 = np.trace(rho@H).real
    E2 = np.trace(sigma@H).real
    E3 = np.trace(tao@H).real

    return E1, E2, E3


def second_order_trace_distance(p_candidate, p_target, W, W2):
    """
    
    W: quarter time step
    W2: Half time step
    
    """
    sim = cirq.Simulator(dtype=np.complex128)
    qbs = cirq.GridQubit.rect(1,5)
    
    A = U_full(p_target)
    B = U_full(p_candidate)
    
    Va = get_env_exact(A)
    Vb = get_env_exact(B)

    c1 = build_second_order_time_evo_rdm(
        Full_U_Gate(p_target), 
        Full_V_Gate(Va), 
        Ham(W),
        DoubleHam(W2)
        )

    c2 = build_no_time_evo_rdm(Full_U_Gate(p_candidate), Full_V_Gate(Vb))

    ρ = sim.simulate(c1).density_matrix_of(qbs[3:5])
    σ = sim.simulate(c2).density_matrix_of(qbs[1:3])

    return np.linalg.norm(ρ - σ, ord = 1)


def adjusted_second_order_trace_distance(p_candidate, p_target, W, W2, H, λ=0.1):
    """
    Add a term into the time evolution cost function which adjusts for the energy drift and biases towards constant energy evolution
    """

    sim = cirq.Simulator(dtype=np.complex128)
    qbs = cirq.GridQubit.rect(1,5)
    
    A = U_full(p_target)
    B = U_full(p_candidate)
    
    Va = get_env_exact(A)
    Vb = get_env_exact(B)

    c1 = build_second_order_time_evo_rdm(
        Full_U_Gate(p_target), 
        Full_V_Gate(Va), 
        Ham(W),
        DoubleHam(W2)
        )

    c2 = build_no_time_evo_rdm(Full_U_Gate(p_candidate), Full_V_Gate(Vb))

    ρ = sim.simulate(c1).density_matrix_of(qbs[3:5])
    σ = sim.simulate(c2).density_matrix_of(qbs[1:3])

    traceDistance = np.linalg.norm(ρ - σ, ord = 1)

    energyCandidate = np.trace(ρ@H).real
    energyTarget = np.trace(σ@H).real

    return traceDistance + λ*np.abs(energyCandidate - energyTarget)


def rdm_trace_distance(p_candidate, p_target, W):
    sim = cirq.Simulator(dtype=np.complex128)
    qbs = cirq.GridQubit.rect(1,5)
    A = U_full(p_target)
    B = U_full(p_candidate)
    Va = get_env_exact(A)
    Vb = get_env_exact(B)

    c1 = build_time_evo_rdm(Full_U_Gate(p_target), Full_V_Gate(Va), Ham(W))

    c2 = build_no_time_evo_rdm(Full_U_Gate(p_candidate), Full_V_Gate(Vb))

    ρ = sim.simulate(c1).density_matrix_of(qbs[2:4])
    σ = sim.simulate(c2).density_matrix_of(qbs[1:3])

    return np.linalg.norm(ρ - σ, ord = 1)


def rdm_time_evolve_step(p_target, W):

    cost_func = partial(
        rdm_trace_distance,
        p_target = p_target,
        W = W
    )

    def show(x):
        print(np.round(cost_func(x),8))
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")


    res = minimize(
        cost_func, p_target, method = "L-BFGS-B", callback=show, tol=1e-8, options = {"disp":False, 'maxiter':15000}
    )

    return res.x


def rdm_time_evolve_step_second_order(p_target, W,W2):

    cost_func = partial(
        second_order_trace_distance,
        p_target = p_target,
        W = W,
        W2 = W2
    )

    def show(x):
        print(np.round(cost_func(x),8))
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")


    res = minimize(
        cost_func, p_target, method = "Nelder-Mead", callback=show, options = {"disp":False, 'maxiter':15000, 'ftol':1e-10, 'xatol':1e-10}
    )

    return res.x


def rdm_time_evolve_step_second_order_energy_bias(p_target, W, W2, H, λ = 0.1):
    
    cost_func = partial(
        adjusted_second_order_trace_distance,
        p_target=p_target,
        W = W,
        W2= W2,
        H = H,
        λ = λ
    )

    def show(x):
        print(np.round(cost_func(x),8))
        sys.stdout.write("\033[F")
        sys.stdout.write("\033[K")


    res = minimize(
        cost_func, p_target, method = "Nelder-Mead", callback=show, options = {"disp":False, 'maxiter':15000, 'ftol':1e-4, 'xatol':1e-4}
    )

    return res.x


class StepParams:
    def __init__(self, steps, dt, fname):
        self.steps = steps
        self.dt = dt
        self.fname = fname


def find_energy_drift_v2():
    H = Hamiltonian({"ZZ":-1, "Z":1}).to_matrices(2)[0]
    random_seed = 500
    np.random.seed(random_seed)

    params = np.random.rand(15)

    steps = [
        StepParams(100, 1e-2, "dt_1e2"),
        StepParams(200, 5e-3, "dt_5e3"),
        StepParams(1000, 1e-3, "dt_1e3")
    ]
    
    energies = []
    for sp in steps:
        steps = sp.steps
        dt = sp.dt
        fname = sp.fname

        W = expm(-1j * H * dt )
        E0 = energy_from_rdm_params(params, H)
        
        for _ in tqdm.tqdm(range(steps)):
            E = energy_from_rdm_params(params, H)
            energies.append(E)
            print(f"Energy Drift: {np.round((E-E0).real, 4)}")

            params = rdm_time_evolve_step(params, W)

        plt.plot(energies)
        plt.savefig(f"experiments/1d/figs/rdm_{fname}_energy_drift_double_time_evo_op.png")
        plt.show()


def find_energy_drift_second_order():
    H = Hamiltonian({"ZZ":1, "X":0.5}).to_matrices(2)[0]
    random_seed = 500
    np.random.seed(random_seed)

    params = np.random.rand(15)

    steps = [
        StepParams(200, 1e-2, "dt_1e2"),
    ]
    
    energies = []
    for sp in steps:
        num_steps = sp.steps
        dt = sp.dt

        W1 = expm(-1j * H * dt/4 )
        W2 = expm(-1j * H * dt/2 )
        
        allParams = []

        for _ in tqdm.tqdm(range(num_steps)):
            allParams.append(params)
            E = energy_from_rdm_params(params, H)
            energies.append(E)
            new_params = rdm_time_evolve_step_second_order_energy_bias(params, W1, W2, H, λ=1.0)
            params = new_params
            
        np.save("allParams.npy", allParams)
        plt.plot(energies)
        plt.ylim(0,1)
        plt.show()


def single_operator_from_params(p, O):
    V = get_env_exact(U_full(p))
    c1 = build_no_time_evo_rdm(Full_U_Gate(p), Full_V_Gate(V)) 
    qubits = cirq.GridQubit.rect(1,5)
    sim = cirq.Simulator(dtype = np.complex128)

    rho = sim.simulate(c1).density_matrix_of(qubits[1:2])

    return np.trace(rho@O).real


def operator_from_files(filename, Op):
    allParams = np.load(filename)
    
    opExpVals = []
    for p in allParams:
        opExpVals.append(single_operator_from_params(p, Op))

    return opExpVals


def plot_energy_change():
    random_seed = 500
    np.random.seed(random_seed)
    params = np.random.rand(15)

    time_ev = []
    no_time_ev = []
    second_order = []

    dts = np.linspace(0,0.005*20,100)
    H = Hamiltonian({"ZZ":1, "X":0.5}).to_matrices(2)[0]
    for dt in dts:

        E1, E2, E3 = compare_energy_change_trotter(dt, H, params)
        time_ev.append(E2)
        no_time_ev.append(E1)
        second_order.append(E3)


    plt.plot(dts, time_ev, 'r', label = 'first_order')
    plt.plot(dts, no_time_ev, 'b', label = 'no_time_evo')
    plt.plot(dts, second_order, 'g', label = 'second_order')

    plt.legend()
    plt.show()


def no_trotter_rdm(p):
    sim = cirq.Simulator(dtype=np.complex128)
    qbs = cirq.GridQubit.rect(1,5)
    A = U_full(p)
    Va = get_env_exact(A)

    c1 = build_no_time_evo_rdm(Full_U_Gate(p), Full_V_Gate(Va))

    ρ = sim.simulate(c1).density_matrix_of(qbs[1:3])

    return ρ


def first_order_rdm(p, W):
    sim = cirq.Simulator(dtype=np.complex128)
    qbs = cirq.GridQubit.rect(1,5)
    A = U_full(p)
    Va = get_env_exact(A)

    c1 = build_time_evo_rdm(Full_U_Gate(p), Full_V_Gate(Va), Ham(W))


    ρ = sim.simulate(c1).density_matrix_of(qbs[2:4])

    return ρ


def second_order_rdm(p, W1, W2):
    """
    
    W1: quarter time step
    W2: half time step
    """
    sim = cirq.Simulator(dtype=np.complex128)
    qbs = cirq.GridQubit.rect(1,5)
    
    A = U_full(p)
    
    Va = get_env_exact(A)

    c1 = build_second_order_time_evo_rdm(
        Full_U_Gate(p), 
        Full_V_Gate(Va), 
        Ham(W1),
        DoubleHam(W2)
        )


    ρ = sim.simulate(c1).density_matrix_of(qbs[3:5])

    return ρ


def trace_distance(ρ, σ):
    return np.linalg.norm(ρ - σ, ord = 1)


def trace_distance_v2(ρ, σ):
    return np.real(
        np.trace(ρ @ ρ) + np.trace(σ @ σ) - 2 * np.real(np.trace(ρ @ σ))
    )


def looking_at_everything():
    random_seed = 500
    np.random.seed(random_seed)
    initParams = np.random.rand(15)
    H = Hamiltonian({"ZZ":1, "X":0.5}).to_matrices(2)[0]

    dt = 1e-2
    W1 = expm(-1j * H * dt/4)
    W2 = expm(-1j * H * dt/2)

    initEnergy = energy_from_rdm_params(initParams, H)
    initRDM = no_trotter_rdm(initParams)

    firstOrderRDM = first_order_rdm(initParams, W2)
    firstOrderEnergy = np.trace(firstOrderRDM @ H).real

    secondOrderRDM = second_order_rdm(initParams, W1, W2)
    secondOrderEnergy = np.trace(secondOrderRDM @ H).real

    # now evolve by a single step 
    newParams = rdm_time_evolve_step_second_order(initParams, W1, W2)
    newRDM = no_trotter_rdm(newParams)

    newEnergy = np.trace(newRDM @ H).real

    # compare the cost functions for these things
    traceDistanceInitNew = trace_distance(initRDM, newRDM)
    traceDistanceNewFirstOrder = trace_distance(newRDM, firstOrderRDM)
    traceDistanceNewSecondOrder = trace_distance(newRDM, secondOrderRDM)
    traceDistanceInitSecondOrder = trace_distance(initRDM, secondOrderRDM)
    traceDistanceInitFirstOrder = trace_distance(initRDM, firstOrderRDM)

    # compare the energies
    ΔEInitNew = np.abs(newEnergy - initEnergy)
    ΔEInitSecond = np.abs(initEnergy - secondOrderEnergy)
    ΔEInitFirst = np.abs(initEnergy - firstOrderEnergy)
    ΔENewFirst = np.abs(newEnergy - firstOrderEnergy)
    ΔENewSecond = np.abs(newEnergy - secondOrderEnergy)


    resultString = f"""

        ---------------------------
        Sanity Check:

        Two Energy Measures Diff: {initEnergy - np.trace(initRDM @ H).real}

        ---------------------------
        Energy Results:

        Initial Energy: {initEnergy}
        First Order Energy: {firstOrderEnergy}
        Second Order Energy: {secondOrderEnergy}
        Optimized Energy: {newEnergy}
        
        -----------------------------
        Trace Distances:

        New <> Init: {traceDistanceInitNew}
        New <> First Order: {traceDistanceNewFirstOrder}
        New <> Second Order: {traceDistanceNewSecondOrder}
        Init <> First Order: {traceDistanceInitFirstOrder}
        Init <> Second Order: {traceDistanceInitSecondOrder}

        ------------------------------
        Energy Differences:

        Init <> New: {ΔEInitNew}
        Init <> First: {ΔEInitFirst}
        Init <> Second: {ΔEInitSecond}

        New <> First: {ΔENewFirst}
        New <> Second: {ΔENewSecond}
    """

    print(resultString)


def energies_from_file(filename, H):
    allParams = np.load(filename)
    
    energies = []
    for p in allParams:
        energies.append(energy_from_rdm_params(p, H))

    return energies




if __name__ == "__main__":
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

    H = Hamiltonian({"ZZ":1, "X":0.5}).to_matrices(2)[0]

    filename = "/home/jamie/work/string_bonds/experiments/1d/allParams.npy"
    
    energies = energies_from_file(filename, H)
    expX = operator_from_files(filename, X)
    expY = operator_from_files(filename, Y)
    expZ = operator_from_files(filename, Z)

    fig, (ax1, ax2) = plt.subplots(2,1)

    ax1.plot(energies)
    ax1.set_ylim(0.05,0.06)
    ax1.title.set_text('Energy Drift')

    ax2.title.set_text('Expectation Values')
    ax2.plot(expX, 'r', label = "⟨X⟩")
    ax2.plot(expY, 'y', label = "⟨Y⟩")
    ax2.plot(expZ, 'b', label = '⟨Z⟩')
    ax2.legend()

    plt.tight_layout()
    plt.show()



