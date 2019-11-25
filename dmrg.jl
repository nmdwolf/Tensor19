using KrylovKit, LinearAlgebra

function iDMRG(multi::Integer=2, bond::Integer=16, max_iter::Integer=100, verbosity::Integer=2)

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
        reshape(H, multi, multi, multi, multi)
    end

    function ApplyHeff(x)
        A = reshape(x, chi * p, :)
        env_ = reshape(L[end], chi * p, :)
        x = env_ * A + A * adjoint(env_)
        x = reshape(x, chi, p, chi, p)
        A = permutedims(reshape(A, chi, p, chi, p), [1 3 2 4])
        h = reshape(A, chi * chi, :) * permutedims(reshape(H, p*p, p*p), [2 1])
        x += permutedims(reshape(h, chi, chi, p, p), [1 3 2 4])
        x = x[:]
    end

    function UpdateHeff(A)
        sizes = size(A)
        Mp = sizes[1] * sizes[2]
        A_ = reshape(A, Mp, :)
        env = adjoint(A_) * reshape(L[end], Mp, Mp) * A_
        id = Diagonal(ones(p))
        env = kron(id, env)
        env = reshape(env, chi, p, chi, p)

        A = reshape(A, :, p * chi)
        B = reshape(adjoint(A) * A, p, chi, p, chi)
        B = reshape(permutedims(B, [2 4 1 3]), chi * chi, p * p)
        h = reshape(permutedims(H, [1 3 2 4]), p * p, p * p)
        env += permutedims(reshape(B * h, chi, chi, p, p), [1 3 2 4])
    end

    function Truncate(A, D)
        u, s, v = svd(reshape(A, chi * p, :))
        cut = length(s)
        if cut > D
            cut = D
        end
        u = u[:, 1:cut]
        reshape(u, chi, p, :), dot(s[cut:end-1], s[cut:end-1])
    end

    H = HeisenbergInteraction()
    p = size(H, 1)

    A = ones(1)
    chi = size(A)[end]
    L = [zeros(chi, p, chi, p)]
    E = 0
    Etot = 0
    dE = 0
    trunc = 0

    for i in 1:max_iter
        d, w = eigsolve(ApplyHeff, chi*p*chi*p)
        A, trunc = Truncate(w[1], bond)
        chi = size(A)[end]
        push!(L, UpdateHeff(A))
        E_ = (d[1] - Etot) / 2
        dE = E - E_
        E = E_
        Etot = d[1]

        if verbosity >= 2
            println("Iter ", i, " >>  M: ", chi, "  E:", E, "  dE: ", dE, "  trunc:", trunc);
        end
    end

    if verbosity >= 1
        println("Final >> M: ", chi, "  E: ", E, "  dE: ", dE, "  trunc: ", trunc);
    end

end

#=
iDMRG()
=#
