using LinearAlgebra, KrylovKit

function VUMPS(multi::Integer=2, bond::Integer=16, max_iter::Integer=100, tol=1e-10, verbosity::Integer=2, canon::Bool=true)

    #=
    All global variables: Mixed orthonormal tensors, dimensions and environments
    =#
    al, ar, ac, c = 0, 0, 0, 0
    Hl, Hr, H = 0, 0, 0
    p, chi = 0, bond

    function dagger(A)
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
        result = Array{ComplexF64}(undef,chi, p * p, chi)
        AA = reshape(AA, chi, p * p, chi)
        NN = reshape(H, p * p, :)

        for i in 1:chi
            result[i,:,:] = NN * AA[i,:,:]
        end

        return result
    end

    function HAc(x)
        result = vec(Hl * reshape(x, chi, :))
        result += vec(reshape(x, :, chi) * transpose(Hr))

        LL = reshape(Al(), :, chi) * reshape(x, chi, :)
        LL = H_2site(LL)

        result += vec(adjoint(reshape(Al(), chi * p, :)) * reshape(LL, chi * p, :))

        RR = reshape(x, :, chi) * reshape(Ar(), chi, :)
        RR = H_2site(RR)

        result += vec(reshape(RR, :, chi * p) * adjoint(reshape(Ar(), :, chi * p)))
        return result
    end

    function Hc(x)
        x = reshape(x, chi, chi)
        result = vec(Hl * x)
        result += vec(x * transpose(Hr))

        C1 = reshape(Al(), :, chi) * x * reshape(Ar(), chi, :)
        C1 = H_2site(C1)

        C3 = reshape(C1, :, chi * p) * adjoint(reshape(Ar(), :, chi * p))
        result += vec(adjoint(reshape(Al(), chi * p, :)) * reshape(C3, chi * p, :))
        return result
    end

    function MakeHeff(A, C, tol=1e-14)
        function P_NullSpace(x)
            #=
            Projecting x on the nullspace of 1 - T
            =#
            x = reshape(x, chi, chi)
            return tr(adjoint(C) * x * C) * Diagonal(ones(chi))
        end

        function Transfer(x)
            #=
            Doing (1 - (T - P)) @ x
            =#
            x = reshape(x, chi, chi)
            res = vec(deepcopy(x))
            res += vec(P_NullSpace(x))
            temp = x * reshape(A, chi, :)
            res -= vec(adjoint(reshape(A, :, chi)) * reshape(temp, :, chi))

            return res
        end

        AA = reshape(A, :, chi) * reshape(A, chi, :)
        HAA = H_2site(AA)

        h = adjoint(reshape(AA, :, chi)) * reshape(HAA, :, chi)

        r, info = linsolve(Transfer, vec(h - P_NullSpace(h)))

        r = reshape(r, chi, chi)
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
        AlHcc = vec(reshape(Al(), :, chi) * reshape(Hcc, chi, :))
        E, norm(HAcAc - 2 * AlHcc) / (2 * abs(E))
    end

    function MakeCanonical(AR, C_in, tol=1e-14)
        A = deepcopy(AR)
        c = Diagonal(ones(chi))

        diff = 1
        iterations = 1
        while diff > tol
            function Transfer(x)
                xA = reshape(x, chi, chi) * reshape(A, chi, :)
                return vec(adjoint(reshape(A, :, chi)) * reshape(xA, :, chi))
            end

            iterations += 1
            d, w = eigsolve(Transfer, chi * chi)
            F = svd(reshape(w[1], chi, chi))
            sqrt_eps = sqrt(eps(1.0))
            s = max.(sqrt.(F.S), sqrt_eps)
            s = s / norm(s)
            c1 = Diagonal(s) * F.Vt
            c1_inv = F.V * (Diagonal(s)^(-1))
            A = c1 * reshape(A, chi, :)
            A = reshape(A, :, chi) * c1_inv
            A = (A / norm(A)) * sqrt(chi)

            c = c1 * c
            c = c / norm(c)
            diff = norm(reshape(w[1], chi, chi) - Diagonal(ones(chi)) * w[1][1])
        end
        A, c / norm(c), [iterations, diff]
    end

    function set_uMPS(AC, C_in, canon::Bool=true, tol=1e-14)
        if canon
            F = svd(reshape(AC, chi, :))
            uar = F.U * F.Vt
            F = svd(reshape(C_in, chi, chi))
            ucr = F.U * F.Vt
            ar = adjoint(ucr) *  uar
            al, c, info = MakeCanonical(Ar(), C_in, tol)
            ac = C() * reshape(Ar(), chi, :)
        else
            c = C
            F = svd(reshape(AC, chi, :))
            uar = F.U * F.Vt
            F = svd(reshape(C, chi, chi))
            ucr = F.U * F.Vt
            F = svd(reshape(AC, :, chi))
            ual = F.U * F.Vt
            F = svd(reshape(C, chi, chi))
            ucl = F.U * F.Vt
            ar = ual * adjoint(ucl)
            al = adjoint(ucr) * uar
            info = [0]
        end
        return info
    end

    function Ar()
        return reshape(ar, chi, p, chi)
    end

    function Al()
        return reshape(al, chi, p, chi)
    end

    function C()
        return reshape(c, chi, chi)
    end

    function Ac()
        return reshape(ac, chi, p, chi)
    end

    function print_info(i, energy, error, ctol, w1, w2, canon_info)
        # println("Iter: ", i, " E: ", energy, " Error: ", error, "tol: ", ctol, " HAc: ", w1, " Hc: ", w2, " c_its: ", canon_info)
        println("Iter: ", i, " E: ", energy, " Error: ", error)
    end

    H = four_site(HeisenbergInteraction(multi))
    p = size(H, 1)

    ac = randn(ComplexF64, chi, p, chi)
    c = randn(ComplexF64, chi, chi)
    ac = ac / norm(ac)
    c = c / norm(c)

    ctol = 1e-3
    error = 1
    canon_info = set_uMPS(ac, c, canon, ctol)

    Hl, ~ = MakeHl(ctol)
    Hr, ~ = MakeHr(ctol)

    for i in 1:max_iter
        ctol = max(min(1e-3, 1e-3 * error), 1e-15)
        if ctol ^ 2 > 1e-16
            etol = ctol ^ 2
        else
            etol = 0
        end

        d1, w1, ~ = eigsolve(HAc, vec(Ac()), 1, :SR)
        d2, w2, ~ = eigsolve(Hc, vec(C()), 1, :SR)

        canon_info = set_uMPS(w1[1], w2[1], canon, ctol)
        Hl, ~ = MakeHl(ctol)
        Hr, ~ = MakeHr(ctol)
        energy, error = current_energy_and_error()

        if error < tol
            break
        end

        if verbosity >= 2
            print_info(i, energy, error, ctol, d1[1], d2[1], canon_info[1])
        end
    end

    if verbosity >= 1
        print_info(i, energy, error, ctol, d1[1], d2[1], canon_info[1])
    end
end

#=
VUMPS()
=#
