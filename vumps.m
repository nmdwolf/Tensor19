function VUMPS(multi, bond, max_iter, tol, verbosity, canon)

    % All global variables: Mixed orthonormal tensors, dimensions and environments
    [al, ar, ac, c] = deal(0, 0, 0, 0);
    [Hl, Hr, H] = deal(0, 0, 0);
    [p, M] = deal(0, bond);

    function n=Norm(x)
        % Helper function to compute Frobenius norms.
        % This method was added because the standard norm() method in MatLab
        % computes the infinity-norm (maximal singular value) instead of the 2-norm.

        n=norm(x, 'fro');
    end

    function A=dagger(A)
        % Helper function to compute adjoints of multi-dimensional arrays.
        A = conj(permute(A, flip(1:length(size(A)))));
    end

    function [Sz,Sp,Sm]=S_operators(multi)
        % Returns the Sz, S+ and S- operators in for a spin.
        % The operators are represented in the Sz basis: (-j, -j + 1, ..., j)
        %
        % Args:
        %    multipl: defines which multiplicity the total spin of the site has.
        %    Thus specifies j as `j = (multipl - 1) / 2`

        if nargin == 0
            j = 1/2;
            multi = 2;
        else
            j = (multi - 1) / 2;
        end
        m = (0:multi-1) - j;

        Sz = diag(m);
        Sp = zeros(length(m));
        temp = sqrt((j - m) .* (j + m + 1));
        Sp(2:(multi+1):end) = temp(1:end-1);

        Sm = Sp.';
    end

    function H=HeisenbergInteraction(multi)
        % Returns Heisenberg interaction between two sites.
        %
        % This is given by:
        %    1/2 * (S_1^+ S_2^- + S_1^- S_2^+) + S_1^z S_2^z
        %
        % Interaction is given in a dense matrix:
        %    Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|

        if nargin == 0
            multi = 2;
        end

        [Sz, Sp, Sm] = S_operators(multi);
        H = 0.5 * (kron(Sp, Sm) + kron(Sm, Sp)) + kron(Sz, Sz);
        H = reshape(H, multi, multi, multi, multi);
    end

    function H=IsingInteraction(multi, J)
        % Returns Ising interaction between two sites.
        %
        % This is given by:
        %    1/2 * (S_1^+  + S_1^- + S_2^+ + S_2^-) + J * S_1^z S_2^z
        %
        % Interaction is given in a dense matrix:
        %    Σ H_{1', 2', 1, 2} |1'〉|2'〉〈1|〈2|

        if nargin == 0
            J = 4;
            multi = 2;
        elseif nargin == 1
            J = 4;
        end

        [Sz, Sp, Sm] = S_operators(multi);
        unity = eye(size(Sz, 1));
        H = 0.5 * (kron(Sp, unity) + kron(Sm, unity) + kron(unity, Sp) + kron(unity, Sm)) + J * kron(Sz, Sz);
        H = reshape(H, multi, multi, multi, multi);
    end

    function result=H_2site(AA)
        % Executes the nearest neighbour interaction on a two-site tensor
        result = zeros(M, p * p, M);
        AA = reshape(AA, M, p * p, M);
        NN = reshape(H, p * p, []);

        for i=1:M
            result(i,:,:) = NN * squeeze(AA(i,:,:));
        end
    end

    function result=HAc(x)
        res = Hl * reshape(x, M, []);
        result = res(:);
        res = reshape(x, [], M) * Hr.';
        result = result + res(:);

        LL = reshape(Al(), [], M) * reshape(x, M, []);
        LL = H_2site(LL);

        res = reshape(Al(), M * p, [])' * reshape(LL, M * p, []);
        result = result + res(:);

        RR = reshape(x, [], M) * reshape(Ar(), M, []);
        RR = H_2site(RR);

        res = reshape(RR, [], M * p) * reshape(Ar(), [], M * p)';
        result = result + res(:);
    end

    function result=Hc(x)
        x = reshape(x, M, M);
        res = Hl * x;
        res2 = x * Hr.';
        result = res(:) + res2(:);

        C1 = reshape(Al(), [], M) * x * reshape(Ar(), M, []);
        C1 = H_2site(C1);

        C3 = reshape(C1, [], M * p) * reshape(Ar(), [], M * p)';
        res = reshape(Al(), M * p, [])' * reshape(C3, M * p, []);
        result = result + res(:);
    end

    function [result, info]=MakeHeff(A, C, tol)
        if nargin == 2
            tol = 1e-14;
        end

        function result=P_NullSpace(x)
            % Projecting x on the nullspace of 1 - T

            x = reshape(x, M, M);
            result = trace(C' * x * C) * eye(M);
        end

        function result=Transfer(x)
            % Doing (1 - (T - P)) @ x

            x = reshape(x, M, M);
            result = x(:);
            res = P_NullSpace(x);
            result = result + res(:);
            tmp = x * reshape(A, M, []);
            res = reshape(A, [], M)' * reshape(tmp, [], M);
            result = result - res(:);
        end

        AA = reshape(A, [], M) * reshape(A, M, []);
        HAA = H_2site(AA);
        h = reshape(AA, [], M)' * reshape(HAA, [], M);

        temp = h - P_NullSpace(h);
        [r, info] = bicgstab(@Transfer, temp(:));

        r = reshape(r, M, M);
        result = r - P_NullSpace(r);
    end

    function [result, info]=MakeHl(tol)
        [result, info] = MakeHeff(Al(), C(), tol);
        if info ~= 0
            display('Making of left environment did not converge.');
        end
    end

    function [result, info]=MakeHr(tol)
        H = dagger(H);
        [result, info] = MakeHeff(dagger(Ar()), dagger(C()), tol);
        H = dagger(H);
        if info ~= 0
            display('Making of right environment did not converge.');
        end
        result = conj(result);
    end

    function H=four_site(h)
        % Transforms the two site interaction to an equivalent four-site
        % interaction such that we can do `two site` optimization which is actually
        % four sites in a time.

        pd = size(h, 1);
        id = eye(pd * pd);
        h = reshape(h, pd * pd, []);
        h2 = 0.5 * reshape(kron(h, id), pd * pd, pd * pd, pd * pd, pd * pd);
        h2 = h2 + 0.5 * reshape(kron(id, h), pd * pd, pd * pd, pd * pd, pd * pd);
        id = eye(pd);
        htemp = reshape(kron(id, h), pd * pd * pd, pd * pd * pd);
        H = (h2 + reshape(kron(htemp, id), pd * pd, pd * pd, pd * pd, pd * pd)) / 2;
    end

    function [energy, error]=current_energy_and_error()
        % Calculates the energy and estimated error of the current uMPS
        %
        % The energy is calculated as the expectation value of Hc for c.
        %
        % The error is calculated as ||HAc @ Ac - 2 * Hc @ c||_frobenius,
        % which should be zero in the fixed point (i.e. when Ac, Al, Ar and c are
        % consistent with each other and Ac and c are eigenstates of HAc and Hc).

        HAcAc = HAc(Ac());
        Hcc = Hc(C());
        c_ = C();
        energy = real(dot(c_(:),Hcc));
        AlHcc = reshape(Al(), [], M) * reshape(Hcc, M, []);
        error = Norm(HAcAc - 2 * AlHcc(:)) / (2 * abs(energy));
    end

    function [A, c, info]=MakeCanonical(AR, C_in, tol)
        if nargin == 2
            tol = 1e-14;
        end

        A = AR;
        c = eye(M);

        diff = 1;
        iterations = 1;

        function result=Transfer(x)
            xA = reshape(x, M, M) * reshape(A, M, []);
            result = reshape(A, [], M)' * reshape(xA, [], M);
            result = result(:);
        end

        while diff > tol
            iterations = iterations + 1;
            [w, d] = eigs(@Transfer, M * M);
            [u, s, v] = svd(reshape(w(:, 1), M, M));
            sqrt_eps = sqrt(eps(1.0));
            s = diag(s);
            s(:) = max(sqrt(s(:)), sqrt_eps);
            s = s / Norm(s);
            c1 = diag(s) * v';
            c1_inv = v * diag(s.^(-1));
            A = c1 * reshape(A, M, []);
            A = reshape(A, [], M) * c1_inv;
            A = (A / Norm(A)) * sqrt(M);

            c = c1 * c;
            c = c / Norm(c);
            diff = Norm(reshape(w(:, 1), M, M) - eye(M) * w(1));
        end
        c = c / Norm(c);
        info = [iterations, diff];
    end

    function info=set_uMPS(AC, C_in, canon, tol)
        if canon
            [uar, ~] = poldec2(reshape(AC, M, []));
            [ucr, ~] = poldec2(reshape(C_in, M, M));
            ar = ucr' *  uar;
            [al, c, info] = MakeCanonical(Ar(), C_in, tol);
            ac = C() * reshape(Ar(), M, []);
        else
            c = C_in;
            [uar, ~] = poldec2(reshape(AC, M, []));
            [ucr, ~] = poldec2(reshape(C, M, M));
            [ual, ~] = poldec(reshape(AC, [], M));
            [ucl, ~] = poldec(reshape(C, M, M));
            ar = ual * ucl';
            al = ucr' * uar;
            info = [0];
        end
    end

    function A=Ar()
        A = reshape(ar, M, p, M);
    end

    function A=Al()
        A = reshape(al, M, p, M);
    end

    function C=C()
        C = reshape(c, M, M);
    end

    function A=Ac()
        A = reshape(ac, M, p, M);
    end

    function print_info(i, energy, error, ctol, w1, w2, canon_info)
        % display(['Iter: ', i, ' E: ', energy, ' Error: ', error, 'tol: ', ctol, ' HAc: ', w1, ' Hc: ', w2, ' c_its: ', canon_info])
        display(['Iter: ', num2str(i), ' E: ', num2str(energy), ' Error: ', num2str(error)]);
    end

    H = four_site(HeisenbergInteraction(multi));
%     H = IsingInteraction(multi);
    p = size(H, 1);

    ac = randn(M * p * M, 1);
    c = randn(M * M, 1);
    ac = ac / Norm(ac);
    c = c / Norm(c);

    ctol = 1e-3;
    error = 1;
    canon_info = set_uMPS(Ac(), C(), canon, ctol);

    [Hl, ~] = MakeHl(ctol);
    [Hr, ~] = MakeHr(ctol);

    for ii=1:max_iter
        ctol = max(min(1e-3, 1e-3 * error), 1e-15);
        if ctol ^ 2 > 1e-16
            etol = ctol ^ 2;
        else
            etol = 0;
        end

        [w1, d1, ~] = eigs(@HAc, prod(size(Ac())), 1, 'smallestreal');
        [w2, d2, ~] = eigs(@Hc, prod(size(C())), 1, 'smallestreal');

        canon_info = set_uMPS(w1(:,1), w2(:, 1), canon, ctol);
        [Hl, ~] = MakeHl(ctol);
        [Hr, ~] = MakeHr(ctol);
        [energy, error] = current_energy_and_error();

        if error < tol
            break;
        end

        if verbosity >= 2
            print_info(ii, energy, error, ctol, d1(1), d2(1), canon_info(1));
        end
    end

    if verbosity >= 1
        print_info(ii, energy, error, ctol, d1(1), d2(1), canon_info(1));
    end
end
