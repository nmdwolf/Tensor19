import numpy as np
from numpy.random import rand
from numpy.linalg import norm
from scipy.linalg import polar, svd
from scipy.sparse.linalg import bicgstab, eigs, eigsh, LinearOperator

def MakeCanonical(Ar, c_in=None, tol=1e-14, dtype=np.float64):
    assert len(Ar.shape) == 3
    bond = Ar.shape[-1]
    A = Ar.copy()
    c = np.eye(bond)

    diff = 1
    iterations = 1
    while diff > tol:
        def Transfer(x, A):
            xA = x.reshape(bond, bond) @ A.reshape(bond, -1)
            return (A.reshape(-1, bond).conj().T @ xA.reshape(-1, bond)).ravel()

        iterations += 1
        LO = LinearOperator(
            (bond ** 2,) * 2,
            matvec=lambda x: Transfer(x, A),
            dtype=dtype
        )
        w, v = eigs(LO, k=1, which='LM')
        U, s, Vh = svd(v[:, 0].reshape(bond, bond))
        sqrt_eps = np.sqrt(np.finfo(dtype).eps)
        s = np.array([max(np.sqrt(st), sqrt_eps) for st in s])
        s = s / norm(s)
        c1 = np.diag(s) @ Vh
        c1_inv = Vh.conj().T @ np.diag(1 / s)
        A = c1 @ A.reshape(bond, -1)
        A = A.reshape(-1, bond) @ c1_inv
        A = A / norm(A) * np.sqrt(bond)

        c = c1 @ c
        c = c / norm(c)
        diff = norm(v[:, 0].reshape(bond, bond) - np.eye(bond) * v.flatten()[0])
    return A, c / norm(c), (iterations, diff)


def four_site(NN):
    """Transforms the two site interaction to an equivalent four-site
    interaction such that we can do `two site` optimization which is actually
    four sites in a time.
    """
    dim = NN.shape[0]
    NN = NN.reshape(dim * dim, -1)
    NN2 = 0.5 * np.kron(NN, np.eye(dim ** 2)).reshape((dim ** 2,) * 4)
    NN2 += 0.5 * np.kron(np.eye(dim ** 2), NN).reshape((dim ** 2,) * 4)
    NNtemp = np.kron(np.eye(dim), NN).reshape((dim ** 3,) * 2)
    return (NN2 + np.kron(NNtemp, np.eye(dim)).reshape((dim ** 2,) * 4)) / 2


def S_operators(multi=2):
    """Returns the S+, S-, and Sz operators in for a spin.
    The operators are represented in the Sz basis: (-j, -j + 1, ..., j)

        Args:
            multi: defines which multiplicity the total spin of the site has.
            Thus specifies j as `j = (multi - 1) / 2`
    """
    j = (multi - 1) / 2
    # magnetic quantum number for eacht basis state in the local basis
    m = np.arange(multi) - j

    Sz = np.diag(m)
    Sp = np.zeros(Sz.shape)
    Sp[range(1, multi), range(0, multi - 1)] = \
        np.sqrt((j - m) * (j + m + 1))[:-1]
    return Sp, Sp.T, Sz


def HeisenbergInteraction(multi=2):
    """Returns Heisenberg interaction between two sites.

        This is given by:
            1/2 * (S_1^+ S_2^- + S_1^- S_2^+) + S_1^z S_2^z

        Interaction is given in a dense matrix:
            Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|
    """
    Sp, Sm, Sz = S_operators(multi)
    H = 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp)) + np.kron(Sz, Sz)
    return H.reshape((multi,) * 4)


def IsingInteraction(multi=2, J=4):
    """Returns Ising interaction between two sites.

        This is given by:
            1/2 * (S_1^+  + S_1^- + S_2^+ + S_2^-) + S_1^z S_2^z

        Interaction is given in a dense matrix:
            Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|
    """
    Sp, Sm, Sz = S_operators(multi)
    unity = np.eye(Sp.shape[0])
    H = 0.5 * (
        np.kron(Sp, unity) + np.kron(Sm, unity) +
        np.kron(unity, Sp) + np.kron(unity, Sm)
    ) + J * np.kron(Sz, Sz)

    return H.reshape((multi,) * 4)


def H_2site(NN_interaction, AA):
    """Executes the nearest neighbour interaction on a two-site tensor
    """
    assert len(AA.shape) == 4
    ppdim = AA.shape[1] * AA.shape[2]
    newshape = (AA.shape[0], ppdim, AA.shape[-1])
    result = np.zeros(newshape, dtype=AA.dtype)

    AA = AA.reshape(newshape)
    NN = NN_interaction.reshape(ppdim, ppdim)
    if AA.shape[0] < AA.shape[-1]:
        for i in range(AA.shape[0]):
            result[i] = NN @ AA[i]
    else:
        for i in range(AA.shape[-1]):
            result[:, :, i] = AA[:, :, i] @ NN.T
    return result

class VUMPS:
    """
    For doing VUMPS.

    VUMPS can use both real and complex tensors.

    Attributes:
        NN_interaction: The Nearest neighbour interaction for the hamiltonian
    """
    def __init__(self, NN_interaction=None, pure_real=False):
        """Initializes the VUMPS object.

        Args:
            NN_interaction: The nearest neighbour interaction.

            If None assume Heisenberg interaction.

            This can also be set to a tensor representing the NN interaction.

            For more information how the passed NN interaction should be
            structured, see the HeisenbergInteraction function.
        """
        if NN_interaction is None:
            self.NN_interaction = HeisenbergInteraction()
        else:
            self.NN_interaction = NN_interaction
        self._dtype = np.float64 if pure_real else np.complex128

    @property
    def p(self):
        """The dimension of the local physical basis.
        """
        assert len(self.NN_interaction.shape) == 4
        return self.NN_interaction.shape[0]

    @property
    def Ar(self):
        """Right canonical tensor.

        Ar, Al and c are made properties to be sure they always have the
        expected shape.
        """
        return self._Ar.reshape(self.bond, self.p, self.bond)

    @Ar.setter
    def Ar(self, Ar):
        self._Ar = Ar
        self._Ac = None

    @property
    def Al(self):
        """Left canonical tensor.
        """
        return self._Al.reshape(self.bond, self.p, self.bond)

    @Al.setter
    def Al(self, Al):
        self._Al = Al
        self._Ac = None

    @property
    def c(self):
        """Central tensor.
        """
        try:
            return self._c.reshape(self.bond, self.bond)
        except AttributeError:
            return None

    @c.setter
    def c(self, c):
        self._c = c
        self._Ac = None

    @property
    def Ac(self):
        if self._Ac is None:
            self._Ac = self.c @ self.Ar.reshape(self.bond, -1)
        return self._Ac.reshape(self.bond, self.p, self.bond)

    def current_energy_and_error(self):
        """Calculates the energy and estimated error of the current uMPS

        The energy is calculated as the expectation value of Hc for c.

        The error is calculated as ||HAc @ Ac - 2 * Hc @ c||_frobenius,
        which should be zero in the fixed point (i.e. when Ac, Al, Ar and c are
        consistent with each other and Ac and c are eigenstates of HAc and Hc).
        """
        HAcAc = self.HAc(self.Ac)
        Hcc = self.Hc(self.c)
        E = np.dot(self.c.ravel().conj(), Hcc.ravel()).real
        AlHcc = (self.Al.reshape(-1, self.bond) @ Hcc.reshape(self.bond, -1)).ravel()
        return E, norm(HAcAc - 2 * AlHcc) / (2 * abs(E))

    def H_2site(self, AA):
        return H_2site(self.NN_interaction,
                       AA.reshape(self.bond, self.p, self.p, self.bond))

    def HAc(self, x):
        # left Heff
        result = (self.Hl @ x.reshape(self.bond, -1)).ravel()
        # right Heff
        result += (x.reshape(-1, self.bond) @ self.Hr.T).ravel()

        # first onsite
        LL = self.Al.reshape(-1, self.bond) @ x.reshape(self.bond, -1)
        LL = self.H_2site(LL)

        result += (self.Al.reshape(self.bond * self.p, -1).conj().T @
                   LL.reshape(self.bond * self.p, -1)).ravel()

        # second onsite
        RR = x.reshape(-1, self.bond) @ self.Ar.reshape(self.bond, -1)
        RR = self.H_2site(RR)

        result += (RR.reshape(-1, self.bond * self.p) @
                   self.Ar.reshape(-1, self.bond * self.p).conj().T).ravel()
        return result

    def Hc(self, x):
        x = x.reshape(self.bond, self.bond)
        # left Heff
        result = (self.Hl @ x).ravel()
        # right Heff
        result += (x @ self.Hr.T).ravel()

        # On site
        C1 = self.Al.reshape(-1, self.bond) @ x @ self.Ar.reshape(self.bond, -1)
        C1 = self.H_2site(C1)

        C3 = C1.reshape(-1, self.p * self.bond) @ \
            self.Ar.reshape(-1, self.p * self.bond).conj().T
        result += (self.Al.reshape(self.bond * self.p, -1).conj().T @
                   C3.reshape(self.p * self.bond, -1)).ravel()
        return result

    def MakeHeff(self, A, c, tol=1e-14):
        def P_NullSpace(x):
            """Projecting x on the nullspace of 1 - T
            """
            x = x.reshape(self.bond, self.bond)
            return np.trace(c.conj().T @ x @ c) * np.eye(self.bond)

        def Transfer(x):
            """Doing (1 - (T - P)) @ x
            """
            x = x.reshape(self.bond, self.bond)
            res = x.ravel().copy()
            res += P_NullSpace(x).ravel()
            temp = x @ A.reshape(self.bond, -1)
            res -= (A.reshape(-1, self.bond).conj().T @
                    temp.reshape(-1, self.bond)).ravel()

            return res

        AA = A.reshape(-1, self.bond) @ A.reshape(self.bond, -1)
        HAA = self.H_2site(AA)

        h = AA.reshape(-1, self.bond).conj().T @ HAA.reshape(-1, self.bond)

        LO = LinearOperator((self.bond * self.bond,) * 2,
                            matvec=Transfer,
                            dtype=self._dtype
                            )

        r, info = bicgstab(LO, (h - P_NullSpace(h)).ravel(), tol=tol)

        # return (1 - P) @ result
        r = r.reshape(self.bond, self.bond)
        return r - P_NullSpace(r), info

    def MakeHl(self, tol):
        result, info = self.MakeHeff(self.Al, self.c, tol)
        if info != 0:
            print(f'Making left environment gave {info} as exit code')
        return result, info

    def MakeHr(self, tol):
        self.NN_interaction = self.NN_interaction.conj().transpose()
        result, info = \
            self.MakeHeff(self.Ar.conj().transpose(), self.c.conj().T, tol)
        self.NN_interaction = self.NN_interaction.conj().transpose()
        if info != 0:
            print(f'Making right environment gave {info} as exit code')
        return result.conj(), info

    def set_uMPS(self, Ac, c, canon=True, tol=1e-14):
        if canon:
            uar = polar(Ac.reshape(self.bond, -1), side='left')[0]
            ucr = polar(c.reshape(self.bond, self.bond), side='left')[0]
            self.Ar = ucr.conj().T @ uar
            self.Al, self.c, info = MakeCanonical(self.Ar, c, tol, self._dtype)
        else:
            self.c = c
            uar = polar(Ac.reshape(self.bond, -1), side='left')[0]
            ucr = polar(c.reshape(self.bond, self.bond), side='left')[0]
            ual = polar(Ac.reshape(-1, self.bond), side='right')[0]
            ucl = polar(c.reshape(self.bond, self.bond), side='right')[0]
            self.Al, self.Ar = ual @ ucl.conj().T, ucr.conj().T @ uar
            info = [0]
        return info

    def kernel(self, max_bond=16, max_iter=1000, tol=1e-10, verbosity=2, canon=True):
        def print_info(i, vumps, ctol, w1, w2, canon_info):
            print(
                f'it: {i},\t'
                f'E: {vumps.energy:.16g},\t'
                f'Error: {vumps.error:.3g},\t'
                f'tol: {ctol:.3g},\t'
                f'HAc: {w1:.6g},\t'
                f'Hc: {w2:.6g},\t'
                f'c_its: {canon_info}'
            )

        self.bond = max_bond
        # Random initial Ac and c guess
        if self._dtype == np.complex128:
            Ac = rand(self.bond, self.p, self.bond) + \
                rand(self.bond, self.p, self.bond) * 1j
            c = rand(self.bond, self.bond) + rand(self.bond, self.bond) * 1j
        else:
            Ac = rand(self.bond, self.p, self.bond)
            c = rand(self.bond, self.bond)
        Ac, c = Ac / norm(Ac), c / norm(c)

        ctol, self.error = 1e-3, 1
        canon_info = self.set_uMPS(Ac, c, canon, tol=ctol)
        self.Hl, _ = self.MakeHl(tol=ctol)
        self.Hr, _ = self.MakeHr(tol=ctol)

        for i in range(max_iter):
            ctol = max(min(1e-3, 1e-3 * self.error), 1e-15)
            etol = ctol ** 2 if ctol ** 2 > 1e-16 else 0

            HAc = LinearOperator(
                (self.bond * self.bond * self.p,) * 2,
                matvec=lambda x: self.HAc(x),
                dtype=self._dtype
            )
            Hc = LinearOperator(
                (self.bond * self.bond,) * 2,
                matvec=lambda x: self.Hc(x),
                dtype=self._dtype
            )

            # Solve the two eigenvalues problem for Ac and c
            w1, v1 = eigsh(HAc, v0=self.Ac.ravel(), k=1, which='SA', tol=etol)
            w2, v2 = eigsh(Hc, v0=self.c.ravel(), k=1, which='SA', tol=etol)

            canon_info = self.set_uMPS(v1[:, 0], v2[:, 0], canon, tol=ctol)
            self.Hl, _ = self.MakeHl(tol=ctol)
            self.Hr, _ = self.MakeHr(tol=ctol)
            self.energy, self.error = self.current_energy_and_error()

            if self.error < tol:
                break

            if verbosity >= 2:
                print_info(i, self, ctol, w1[0], w2[0], canon_info[0])

        if verbosity >= 1:
            print_info(i, self, ctol, w1[0], w2[0], canon_info[0])


if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        max_bond = [int(d) for d in argv[1:]]
    else:
        max_bond = [16]

    vumps = VUMPS(NN_interaction=four_site(HeisenbergInteraction()))
    # vumps = VUMPS(NN_interaction=IsingInteraction())
    for d in max_bond:
        vumps.kernel(max_bond=d, max_iter=100, canon=True)
