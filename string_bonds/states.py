from xmps.iMPS import iMPS, Map
from xmps.ncon import ncon
import numpy as np
from numpy.linalg import cholesky
import qmps.tools as tools
from xmps.spin import paulis

np.random.seed(500)
X, Y, Z = paulis(0.5)


def merge(A, B):
    # -A- -B-  ->  -A-B-
    #  |   |        ||
    return (
        np.tensordot(A, B, [2, 1]).transpose([0, 2, 1, 3]).reshape(2 * A.shape[0], 2, 2)
    )


class StringBondState:
    """String Bond states look like stacked MPS, with two rows of MPS unitaries.

      /|  /|  /| <- U1
     /|. /|. /|.
    /|. /|  /|.
    --- --- ---  <- U2
     |   |   |
    """

    def __init__(self, U1, U2):
        """Create a string bond state

        Args:
            U1: The first unitary. This is the one on top.
            U2: The second unitary. This is the one below.
        """
        self.U1 = U1
        self.U2 = U2

    def set_environments(self):
        """String bond states need three environments to calculate all two site expectation values
        (both x and y).
        This function sets them.
        """
        e1 = self.get_e1()
        e2 = self.get_e2(e1)
        e3 = self.get_e3(e1)

        # Now e3
        self._e1, self._e2, self._e3 = e1, e2, e3

    def get_e1(self):
        A1 = tools.unitary_to_tensor(self.U1)
        η1, e1 = Map(A1, A1).right_fixed_point()  # first environment.
        e1 = e1 / np.trace(e1)  # renormalise
        return e1

    def get_e2(self, e1):
        A1 = tools.unitary_to_tensor(self.U1)

        e1_half = cholesky(e1)
        A1 = A1 @ e1_half  # make the centre gauge tensor

        U2 = self.U2.reshape(2, 2, 2, 2)

        A2 = ncon(
            [U2, A1],
            [[-4, -1, 1, -5], [1, -2, -3]],  # make the second order tensor.
        ).reshape(8, 2, 2)

        η2, e2 = Map(A2, A2).right_fixed_point()  # second environment

        e2 = e2 / np.trace(e2)
        return e2

    def get_e3(self, e1):
        A1 = tools.unitary_to_tensor(self.U1)

        AA = merge(A1, A1)
        e1_half = cholesky(e1)
        AA = AA @ e1_half  # make the centre gauge tensor

        U_ = self.U2.reshape(2, 2, 2, 2)
        UU = ncon([U_, U_], [[-1, -3, -5, -7], [-2, -4, -6, -8]]).reshape(4, 4, 4, 4)

        A3 = ncon(
            [UU, AA],
            [[-4, -1, 1, -5], [1, -2, -3]],  # make the second order tensor.
        ).reshape(16, 4, 4)

        η3, e3 = Map(A3, A3).right_fixed_point()  # second environment

        e3 = e3 / np.trace(e3)

        return e3

    @property
    def e1(self):
        if not hasattr(self, "_e1"):
            self.set_environments()
        return self._e1

    @e1.setter
    def e1(self, x):
        self._e1 = x

    @property
    def e2(self):
        if not hasattr(self, "_e2"):
            self.set_environments()
        return self._e2

    @e2.setter
    def e2(self, x):
        self._e2 = x

    @property
    def e3(self):
        if not hasattr(self, "_e3"):
            self.set_environments()
        return self._e3

    @e3.setter
    def e3(self, x):
        self._e3 = x

    def E(self, op):
        """Expectation value of a 1 site operator

        Args:
            op:
        """
        e1, e2 = self.e1, self.e2
        A1 = tools.unitary_to_tensor(self.U1) @ cholesky(e1)  # centre gauge tensor

        U2 = self.U2 @ np.kron(np.eye(2), cholesky(e2))  # centre gauge tensor (unitary)

        U2 = U2
        U2_conj = tools.cT(U2)
        return np.real(
            ncon(
                [
                    U2.reshape(2, 2, 2, 2),
                    op,
                    U2_conj.reshape(2, 2, 2, 2),
                    A1,
                    A1.conj(),
                ],
                [[1, 2, 3, 4], [6, 2], [5, 4, 1, 6], [3, 7, 8], [5, 7, 8]],
            )
        )

    def E2x(self, op):
        """Two site expectation values in the x direction

        Args:
            op: op to get the expectation value of.
        """
        e1, e2 = self.e1, self.e2
        A = tools.unitary_to_tensor(self.U1) @ cholesky(
            e1
        )  # centre gauge tensor (drop the 1 index)

        # this is confusing: this is not self.U1, but rather the first unitary in the diagram,
        # which doesn't have an environment tensor contracted into it
        U1 = self.U2  # First unitary (drop the 2 index)
        U1_conj = tools.cT(U1)

        U2 = self.U2 @ np.kron(np.eye(2), cholesky(e2))  # centre gauge tensor (unitary)
        U2_conj = tools.cT(U2)

        return np.real(
            ncon(
                [
                    U1.reshape(2, 2, 2, 2),
                    U2.reshape(2, 2, 2, 2),
                    op.reshape(2, 2, 2, 2),
                    U1_conj.reshape(2, 2, 2, 2),
                    U2_conj.reshape(2, 2, 2, 2),
                    A,
                    A.conj(),
                    A,
                    A.conj(),
                ],
                [
                    [1, 2, 3, 4],
                    [4, 5, 6, 7],
                    [8, 9, 2, 5],
                    [10, 11, 1, 8],
                    [13, 7, 11, 9],
                    [3, 14, 15],
                    [10, 14, 15],
                    [6, 16, 17],
                    [13, 16, 17],
                ],
            )
        )

    def E2y(self, op):
        """Two site expectation values in the y direction.

        Args:
            op: op to get the ev of.
        """
        e1, e2, e3 = self.e1, self.e2, self.e3
        A1 = tools.unitary_to_tensor(self.U1)

        AA = merge(A1, A1)
        e1_half = cholesky(e1)
        AA = AA @ e1_half  # make the (doubled) centre outer gauge tensor

        U_ = self.U2.reshape(2, 2, 2, 2)
        # doubled centre gauge inner tensor
        UU = ncon([U_, U_], [[-1, -3, -5, -7], [-2, -4, -6, -8]]).reshape(
            16, 16
        ) @ np.kron(np.eye(4), cholesky(e3))

        UU_conj = tools.cT(UU)
        return np.real(  # same as e1!
            ncon(
                [
                    UU.reshape(4, 4, 4, 4),
                    op,
                    UU_conj.reshape(4, 4, 4, 4),
                    AA,
                    AA.conj(),
                ],
                [[1, 2, 3, 4], [6, 2], [5, 4, 1, 6], [3, 7, 8], [5, 7, 8]],
            )
        )

    def rhox(self):
        """rhox: 2x1 density matrix"""
        e1, e2 = self.e1, self.e2
        A = tools.unitary_to_tensor(self.U1) @ cholesky(
            e1
        )  # centre gauge tensor (drop the 1 index

        # this is confusing: this is not self.U1, but rather the first unitary in the diagram,
        # which doesn't have an environment tensor contracted into it
        U1 = self.U2  # First unitary (drop the 2 index)
        U1_conj = tools.cT(U1)

        U2 = self.U2 @ np.kron(np.eye(2), cholesky(e2))  # centre gauge tensor (unitary)
        U2_conj = tools.cT(U2)

        half = ncon(
            [U1.reshape(2, 2, 2, 2), U2.reshape(2, 2, 2, 2), A, A],
            [[-3, -1, 1, 2], [2, -2, 3, -4], [1, -7, -8], [3, -5, -6]],
        )
        return ncon(
            [half, half.conj()],
            [[-1, -2, 1, 2, 3, 4, 5, 6], [-3, -4, 1, 2, 3, 4, 5, 6]],
        ).reshape(4, 4)

    def rhoy(self):
        """rhoy: 1x2 density matrix"""
        e1, e2, e3 = self.e1, self.e2, self.e3
        A1 = tools.unitary_to_tensor(self.U1)

        AA = merge(A1, A1)
        e1_half = cholesky(e1)
        AA = AA @ e1_half  # make the (doubled) centre outer gauge tensor

        U_ = self.U2.reshape(2, 2, 2, 2)
        # doubled centre gauge inner tensor
        UU = ncon([U_, U_], [[-1, -3, -5, -7], [-2, -4, -6, -8]]).reshape(
            16, 16
        ) @ np.kron(np.eye(4), cholesky(e3))

        UU_conj = tools.cT(UU)
        half = ncon([UU.reshape(4, 4, 4, 4), AA], [[-2, -1, 1, -3], [1, -4, -5]])
        return ncon([half, half.conj()], [[-1, 1, 2, 3, 4], [-2, 1, 2, 3, 4]])

    def energy(self, opx, opy=None):
        """Energy (density). Measure the expectation value of a two site operator in the x and y directions and add (/2)

        Args:
            op: The local (two site) hamiltonian
        """
        opy = opx if opy is None else opy
        return (self.E2x(opx) + self.E2y(opy)) / 2
