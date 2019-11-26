using LinearAlgebra, KrylovKit

#=
   Variables:
       * multi: Spin multiplicity of every site
       * max_bond: Maximum bond dimension of the states
       * max_iter: Maximum number of iterations for the algorithm
       * verbosity:
            0: Don't print anything.
            1: Print results for the optimization.
            2: Print intermediate result at every even chain length.
=#
function VUMPS(multi::Integer=2, max_bond::Integer=16, max_iter::Integer=100, tol=1e-10, verbosity::Integer=2, canon::Bool=true)

    #=
    All global variables: Mixed orthonormal tensors, dimensions, environments and temporary info
    =#
    al, ar, ac, c = 0, 0, 0, 0
    Hl, Hr, H = 0, 0, 0
    p, bond = 0, max_bond
    energy, error = 0, 0

    function poldec(A)
        # Helper function calculating the polar decomposition in terms of the SVD.

        F = svd(A)
        U = F.U * F.Vt
    end

    function dagger(A)
        # Helper function to compute adjoints of multi-dimensional arrays.

        return conj(permutedims(A, reverse(1:length(size(A)))))
    end

    function S_operators(multi::Integer=2)
        j = (multi - 1) / 2
        m = range(0, length=multi) .- j

        Sz = Diagonal(m)
        Sp = zeros(size(Sz))
        Sp[2:multi+1:end] = sqrt.(((j .- m) .* (j .+ m .+ 1)))[1:end-1]

        Sz, Sp, Sp'
    end

    function HeisenbergInteraction(multi::Integer=2)
        Sz, Sp, Sm = S_operators(multi)
        H = 0.5 * (kron(Sp, Sm) + kron(Sm, Sp)) + kron(Sz, Sz)
        return reshape(H, multi, multi, multi, multi)
    end

    function IsingInteraction(multi::Integer=2, J::Integer=4)
        #=
        Returns Ising interaction between two sites.

        This is given by:
            1/2 * (S_1^+  + S_1^- + S_2^+ + S_2^-) + S_1^z S_2^z

        Interaction is given in a dense matrix:
            Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|
        =#
        Sz, Sp, Sm = S_operators(multi)
        unity = Diagonal(ones(size(Sp)[1]))
        H = 0.5 * (kron(Sp, unity) + kron(Sm, unity) + kron(unity, Sp) + kron(unity, Sm)) + J * kron(Sz, Sz)
        return reshape(H, multi, multi, multi, multi)
    end

    function H_2site(AA)
        #=
        Executes the nearest neighbour interaction on a two-site tensor
        =#
        result = Array{ComplexF64}(undef,bond, p * p, bond)
        AA = reshape(AA, bond, p * p, bond)
        NN = reshape(H, p * p, :)

        for i in 1:bond
            result[i,:,:] = NN * AA[i,:,:]
        end

        return result
    end

    function HAc(x)
        result = vec(Hl * reshape(x, bond, :))
        result += vec(reshape(x, :, bond) * transpose(Hr))

        LL = reshape(Al(), :, bond) * reshape(x, bond, :)
        LL = H_2site(LL)

        result += vec(adjoint(reshape(Al(), bond * p, :)) * reshape(LL, bond * p, :))

        RR = reshape(x, :, bond) * reshape(Ar(), bond, :)
        RR = H_2site(RR)

        result += vec(reshape(RR, :, bond * p) * adjoint(reshape(Ar(), :, bond * p)))
        return result
    end

    function Hc(x)
        x = reshape(x, bond, bond)
        result = vec(Hl * x)
        result += vec(x * transpose(Hr))

        C1 = reshape(Al(), :, bond) * x * reshape(Ar(), bond, :)
        C1 = H_2site(C1)

        C3 = reshape(C1, :, bond * p) * adjoint(reshape(Ar(), :, bond * p))
        result += vec(adjoint(reshape(Al(), bond * p, :)) * reshape(C3, bond * p, :))
        return result
    end

    function MakeHeff(A, C, tol=1e-14)
        function P_NullSpace(x)
            #=
            Projecting x on the nullspace of 1 - T
            =#
            x = reshape(x, bond, bond)
            return tr(adjoint(C) * x * C) * Diagonal(ones(bond))
        end

        function Transfer(x)
            #=
            Doing (1 - (T - P)) @ x
            =#
            x = reshape(x, bond, bond)
            res = vec(deepcopy(x))
            res += vec(P_NullSpace(x))
            temp = x * reshape(A, bond, :)
            res -= vec(adjoint(reshape(A, :, bond)) * reshape(temp, :, bond))

            return res
        end

        AA = reshape(A, :, bond) * reshape(A, bond, :)
        HAA = H_2site(AA)

        h = adjoint(reshape(AA, :, bond)) * reshape(HAA, :, bond)

        r, info = linsolve(Transfer, vec(h - P_NullSpace(h)), tol=tol)

        r = reshape(r, bond, bond)
        r - P_NullSpace(r), info
    end

    function MakeHl(tol)
        result, info = MakeHeff(Al(), C(), tol)
        if info.converged != 1
            println("Making of left environment did not converge after ", info.numiter, " iterations.")
        end
        result, info
    end

    function MakeHr(tol)
        H = dagger(H)
        result, info = MakeHeff(dagger(Ar()), dagger(C()), tol)
        H = dagger(H)
        if info.converged != 1
            println("Making of right environment did not converge after ", info.numiter, " iterations.")
        end
        conj(result), info
    end

    function four_site(h)
        #=
        Transforms the two site interaction to an equivalent four-site
        interaction such that we can do `two site` optimization which is actually
        four sites in a time.
        =#
        pd = size(h, 1)
        id = Diagonal(ones(pd * pd))
        h = reshape(h, pd * pd, :)
        h2 = 0.5 * reshape(kron(h, id), pd * pd, pd * pd, pd * pd, pd * pd)
        h2 += 0.5 * reshape(kron(id, h), pd * pd, pd * pd, pd * pd, pd * pd)
        id = Diagonal(ones(pd))
        htemp = reshape(kron(id, h), pd * pd * pd, pd * pd * pd)
        return (h2 + reshape(kron(htemp, id), pd * pd, pd * pd, pd * pd, pd * pd)) / 2
    end

    function current_energy_and_error()
        #=Calculates the energy and estimated error of the current uMPS

        The energy is calculated as the expectation value of Hc for c.

        The error is calculated as ||HAc @ Ac - 2 * Hc @ c||_frobenius,
        which should be zero in the fixed point (i.e. when Ac, Al, Ar and c are
        consistent with each other and Ac and c are eigenstates of HAc and Hc).
        =#
        HAcAc = HAc(Ac())
        Hcc = Hc(C())
        E = real(dot(vec(C()),Hcc))
        AlHcc = vec(reshape(Al(), :, bond) * reshape(Hcc, bond, :))
        E, norm(HAcAc - 2 * AlHcc) / (2 * abs(E))
    end

    function MakeCanonical(AR, C_in, tol=1e-14)
        A = deepcopy(AR)
        c = Diagonal(ones(bond))

        diff = 1
        iterations = 1
        while diff > tol
            function Transfer(x)
                xA = reshape(x, bond, bond) * reshape(A, bond, :)
                return vec(adjoint(reshape(A, :, bond)) * reshape(xA, :, bond))
            end
            iterations += 1
            d, w = eigsolve(Transfer, bond * bond, 1, tol=tol)
            F = svd(reshape(w[1], bond, bond))
            sqrt_eps = sqrt(eps(1.0))
            s = max.(sqrt.(F.S), sqrt_eps)
            s = s / norm(s)
            c1 = Diagonal(s) * F.Vt
            c1_inv = F.V * (Diagonal(s)^(-1))
            A = c1 * reshape(A, bond, :)
            A = reshape(A, :, bond) * c1_inv
            A = (A / norm(A)) * sqrt(bond)

            c = c1 * c
            c = c / norm(c)
            diff = norm(reshape(w[1], bond, bond) - Diagonal(ones(bond)) * w[1][1])
        end
        A, c / norm(c), [iterations, diff]
    end

    function set_uMPS(AC, C_in, canon::Bool=true, tol=1e-14)
        if canon
            uar = poldec(reshape(AC, bond, :))
            ucr = poldec(reshape(C_in, bond, bond))
            ar = adjoint(ucr) *  uar
            al, c, info = MakeCanonical(Ar(), C_in, tol)
            ac = C() * reshape(Ar(), bond, :)
        else
            c = C_in
            uar = poldec(reshape(AC, bond, :))
            ucr = poldec(reshape(C, bond, bond))
            ar = uar * adjoint(ucr)
            al = adjoint(ucr) * uar
            info = [0]
        end
        return info
    end

    function Ar()
        return reshape(ar, bond, p, bond)
    end

    function Al()
        return reshape(al, bond, p, bond)
    end

    function C()
        return reshape(c, bond, bond)
    end

    function Ac()
        return reshape(ac, bond, p, bond)
    end

    function print_info(i, energy, error, ctol, canon_info)
        println("Iter: ", i, " E: ", energy, " Error: ", error, " tol: ", ctol, " c_its: ", canon_info)
        # println("Iter: ", i, " E: ", energy, " Error: ", error)
    end

    H = four_site(HeisenbergInteraction(multi))
    p = size(H, 1)

    ac = randn(ComplexF64, bond, p, bond)
    c = randn(ComplexF64, bond, bond)
    ac = ac / norm(ac)
    c = c / norm(c)

    ctol = 1e-3
    error = 1
    canon_info = set_uMPS(ac, c, canon, ctol)

    Hl, ~ = MakeHl(ctol)
    Hr, ~ = MakeHr(ctol)

    for i in 1:max_iter
        ctol = max(min(1e-3, 1e-3 * error), 1e-15)

        d1, w1, ~ = eigsolve(HAc, vec(Ac()), 1, :SR)
        d2, w2, ~ = eigsolve(Hc, vec(C()), 1, :SR)

        canon_info = set_uMPS(w1[1], w2[1], canon, ctol)
        Hl, ~ = MakeHl(ctol)
        Hr, ~ = MakeHr(ctol)
        energy, error = current_energy_and_error()

        if error < tol
            println("Algorithm converged:")
            break
        end

        if verbosity >= 2
            print_info(i, energy, error, ctol, canon_info[1])
        end
    end

    if verbosity >= 1
        println("Final values: E: ", energy, " Error: ", error)
    end
end

# VUMPS()
