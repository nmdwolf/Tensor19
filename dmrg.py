# - *- coding: utf- 8 - *-
import numpy as np

class OGDMRG:

    def kernel(self, max_bond=16, max_iter=100, verbosity=2):
        """Execution of the DMRG algorithm.

        Args:
            max_bond: The bond dimension to use for DMRG. The algorithm can choose
            a bond dimension larger than the one specified to avoid truncating
            between renormalized states degenerate in their singular values.

            max_iter: The maximal iterations to use in the DMRG algorithm.

            verbosity: 0: Don't print anything.
                       1: Print results for the optimization.
                       2: Print intermediate result at every even chain length.
        """
        from scipy.sparse.linalg import eigsh, LinearOperator

        for i in range(0, max_iter, 2):
            # even/odd chain length
            for j in range(2):
                H = LinearOperator(((self.bond * self.p) ** 2,) * 2,
                                   matvec=lambda x: self.Heff(x))
                # Diagonalize
                w, v = eigsh(H, k=1, which='SA')
                # Renormalize the basis
                trunc = self.renormalize_basis(v[:, 0], max_bond)
                # Update the effective Hamiltonian
                self.update_Heff()

            # Energy difference between this and the previous even-length chain
            E = (w[0] - self.Etot) / 4
            ΔE, self.E, self.Etot = self.E - E, E, w[0]

            if verbosity >= 2:
                print(f"it {i}:\tM: {self.bond},\tE: {self.E:.12f},\t"
                      f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g}")
        if verbosity >= 1:
            print(f"M: {self.bond},\tE: {self.E:.12f},\t"
                  f"ΔE: {ΔE:.3g},\ttrunc: {trunc:.3g}")

        return self.E

    """
    Attributes:
        NN_interaction: The Nearest neighbour interaction for the hamiltonian
        HA: The current effective Hamiltonian
        E: The current energy per site
        Etot: The current total energy of the system
    """
    def __init__(self, NN_interaction=None, multi=2):
        """Initializes the OGDMRG object.

        Args:
            NN_interaction: The nearest neighbour interaction. If None, a
            Heisenberg interaction is assumed.

            For more information how the passed NN interaction should be
            structured, see the HeisenbergInteraction function.

            multi: If NN_interaction is not None, this argument is ignored.
            If NN_interaction is None, this argument specifies the multiplicity
            of each spin for the Heisenberg interaction.

            E.g. For `multi = 2`, we work with spin-1/2.
                 For `multi = 3`, we work with spin-1.
        """
        if NN_interaction is None:
            self.NN_interaction = OGDMRG.HeisenbergInteraction(multi)
        else:
            self.NN_interaction = NN_interaction

        self.A = np.ones((1, 1))
        self.HA = np.zeros((self.bond, self.p, self.bond, self.p))
        self.E = 0
        self.Etot = 0

    @property
    def bond(self):
        """The current bond dimension used for DMRG.

        It is equal to the last dimension of the current A tensor.
        """
        return self.A.shape[-1]

    @property
    def p(self):
        """The dimension of the local physical basis.
        """
        assert len(self.NN_interaction.shape) == 4
        return self.NN_interaction.shape[0]

    @property
    def A(self):
        """The current A-tensor."""
        return self._A

    @A.setter
    def A(self, A):
        self._A = A

    def S_operators(multi=2):
        """Returns the S+, S-, and Sz operators for a given spin multiplicity.

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
        return Sz, Sp, Sp.T

    def HeisenbergInteraction(multi=2):
        """Returns Heisenberg interaction between two sites.

        This is given by:
            1/2 * (S_1^+ S_2^- + S_1^- S_2^+) + S_1^z S_2^z

        Interaction is given in a dense matrix:
            Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|
        """
        Sz, Sp, Sm = OGDMRG.S_operators(multi)
        H = 0.5 * (np.kron(Sp, Sm) + np.kron(Sm, Sp)) + np.kron(Sz, Sz)
        return H.reshape((multi,) * 4)

    def Heff(self, x):
        """Executing the Effective Hamiltonian on the two-site object `x`.

        The Effective Hamiltonian exists out of:
            * Interactions between left environment and left site
            * Interactions between right environment and right site
            * Interactions between the left and right site

        Returns H_A * x.
        """
        x = x.reshape(self.bond * self.p, -1)
        # Interactions between left environment and left site
        result = self.HA.reshape(self.bond * self.p, -1) @ x
        # Interactions between right environment and right site
        result += x @ self.HA.reshape(self.bond * self.p, -1).T
        # Interactions between left and right site
        x = x.reshape(self.bond, self.p, self.bond, self.p)
        result = result.reshape(self.bond, self.p, self.bond, self.p)
        result += np.einsum('xyij,lirj->lxry', self.NN_interaction, x)

        return result.ravel()

    def update_Heff(self):
        """
        Update the effective Hamiltonian for the new renormalized basis.
        """
        dim = self.A.shape[0] * self.A.shape[1]

        A = self.A.reshape(dim, -1)
        tH = A.T @ self.HA.reshape(dim, dim) @ A
        self.HA = np.kron(tH, np.eye(self.p))
        self.HA = self.HA.reshape(self.bond, self.p, self.bond, self.p)

        A = self.A.reshape(-1, self.p * self.bond)
        B = (A.T @ A).reshape(self.p, self.bond, self.p, self.bond)
        self.HA += np.einsum('ibjc,ikjl->bkcl', B, self.NN_interaction)

    def renormalize_basis(self, A2, max_bond, tol=1e-10):
        """Renormalize the basis.
        """
        from numpy.linalg import svd, qr

        u, s, v = svd(A2.reshape((self.bond * self.p, self.bond * self.p)))

        # Truncating renormalized basis
        if len(s) > max_bond:
            cut = max_bond
        else:
            cut = len(s)
        u = u[:,:cut]

        self.A = u.reshape((self.bond, self.p, -1))
        return s[cut:] @ s[cut:]

if __name__ == '__main__':
    from sys import argv
    if len(argv) > 1:
        max_bond = [int(d) for d in argv[1:]]
    else:
        max_bond = [16]

    ogdmrg = OGDMRG()
    for d in max_bond:
        ogdmrg.kernel(max_bond=d)
