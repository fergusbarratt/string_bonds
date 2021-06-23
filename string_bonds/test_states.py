from states import *
from xmps.Hamiltonians import Hamiltonian


def test_normalised():
    A = iMPS().random(2, 2).left_canonicalise()
    B = iMPS().random(2, 2).left_canonicalise()
    UA = tools.tensor_to_unitary(A[0])
    UB = tools.tensor_to_unitary(B[0])

    state = StringBondState(UA, UB)
    assert np.allclose(state.E(np.eye(2)), 1)
    assert np.allclose(state.E2x(np.eye(4)), 1)
    assert np.allclose(state.E2y(np.eye(4)), 1)
