using KrylovKit, LinearAlgebra

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
function dmrg(multi::Integer=2, max_bond::Integer=16, max_iter::Integer=100, verbosity::Integer=2)

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
        A = reshape(x, bond * p, :)
        env_ = reshape(L, bond * p, :)
        x = env_ * A + A * adjoint(env_)
        x = reshape(x, bond, p, bond, p)
        A = permutedims(reshape(A, bond, p, bond, p), [1 3 2 4])
        h = reshape(A, bond * bond, :) * permutedims(reshape(H, p*p, p*p), [2 1])
        x += permutedims(reshape(h, bond, bond, p, p), [1 3 2 4])
        x = x[:]
    end

    function UpdateHeff(A)
        sizes = size(A)
        dim = sizes[1] * sizes[2]
        A_ = reshape(A, dim, :)
        env = adjoint(A_) * reshape(L, dim, dim) * A_
        id = Diagonal(ones(p))
        env = kron(id, env)
        env = reshape(env, bond, p, bond, p)

        A = reshape(A, :, p * bond)
        B = reshape(adjoint(A) * A, p, bond, p, bond)
        B = reshape(permutedims(B, [2 4 1 3]), bond * bond, p * p)
        h = reshape(permutedims(H, [1 3 2 4]), p * p, p * p)
        env += permutedims(reshape(B * h, bond, bond, p, p), [1 3 2 4])
    end

    function Truncate(A, D)
        u, s, v = svd(reshape(A, bond * p, :))
        cut = length(s)
        if cut > D
            cut = D
        end
        u = u[:, 1:cut]
        reshape(u, bond, p, :), dot(s[cut:end-1], s[cut:end-1])
    end

    H = HeisenbergInteraction()
    p = size(H, 1)

    A = ones(1)
    bond = size(A)[end]
    L = zeros(bond, p, bond, p)
    E = 0
    Etot = 0
    dE = 0
    trunc = 0

    for i in 1:max_iter
        d, w = eigsolve(ApplyHeff, bond*p*bond*p, 1, :SR)
        A, trunc = Truncate(w[1], max_bond)
        bond = size(A)[end]
        L = UpdateHeff(A)
        E_ = (d[1] - Etot) / 2
        dE = E - E_
        E = E_
        Etot = d[1]

        if verbosity >= 2
            println("Iter ", i, " >>  M: ", bond, "  E:", E, "  dE: ", dE, "  trunc:", trunc);
        end
    end

    if verbosity >= 1
        println("Final >> M: ", bond, "  E: ", E, "  dE: ", dE, "  trunc: ", trunc);
    end

end

dmrg()
