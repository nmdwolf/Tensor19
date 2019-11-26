%   Variables:
%       * multi: Spin multiplicity of every site
%       * max_bond: Maximum bond dimension of the states
%       * max_iter: Maximum number of iterations for the algorithm
%       * verbosity:
%            0: Don't print anything.
%            1: Print results for the optimization.
%            2: Print intermediate result at every even chain length.
function vumps(multi, max_bond, max_iter, tol, verbosity, canon)

    % All global variables: Mixed orthonormal tensors, dimensions and environments
    [al, ar, ac, c] = deal(0, 0, 0, 0);
    [Hl, Hr, H] = deal(0, 0, 0);
    [p, bond] = deal(0, max_bond);

    function U = poldec(A)
	% Helper function calculating the polar decomposition in terms of the SVD.
	
        [P, S, Q] = svd(A, 'econ');  % Economy size.
        U = P*Q';
    end
    
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

    function H=four_site(h)
        % Transforms the two site interaction to an equivalent four-site
        % interaction such that we can do `two site` optimization which is actually
        % four sites in a time.

        dim = size(h, 1);
        id = eye(dim * dim);
        h = reshape(h, dim * dim, []);
        h2 = 0.5 * reshape(kron(h, id), dim * dim, dim * dim, dim * dim, dim * dim);
        h2 = h2 + 0.5 * reshape(kron(id, h), dim * dim, dim * dim, dim * dim, dim * dim);
        id = eye(dim);
        htemp = reshape(kron(id, h), dim * dim * dim, dim * dim * dim);
        H = (h2 + reshape(kron(htemp, id), dim * dim, dim * dim, dim * dim, dim * dim)) / 2;
    end

    function result=H_2site(AA)
        % Executes the nearest neighbour interaction on a two-site tensor
        result = zeros(bond, p * p, bond);
        AA = reshape(AA, bond, p * p, bond);
        NN = reshape(H, p * p, []);

        for i=1:bond
            result(i,:,:) = NN * squeeze(AA(i,:,:));
        end
    end

    function result=HAc(x)
        res = Hl * reshape(x, bond, []);
        result = res(:);
        res = reshape(x, [], bond) * Hr.';
        result = result + res(:);

        LL = reshape(Al(), [], bond) * reshape(x, bond, []);
        LL = H_2site(LL);

        res = reshape(Al(), bond * p, [])' * reshape(LL, bond * p, []);
        result = result + res(:);

        RR = reshape(x, [], bond) * reshape(Ar(), bond, []);
        RR = H_2site(RR);

        res = reshape(RR, [], bond * p) * reshape(Ar(), [], bond * p)';
        result = result + res(:);
    end

    function result=Hc(x)
        x = reshape(x, bond, bond);
        res = Hl * x;
        res2 = x * Hr.';
        result = res(:) + res2(:);

        C1 = reshape(Al(), [], bond) * x * reshape(Ar(), bond, []);
        C1 = H_2site(C1);

        C3 = reshape(C1, [], bond * p) * reshape(Ar(), [], bond * p)';
        res = reshape(Al(), bond * p, [])' * reshape(C3, bond * p, []);
        result = result + res(:);
    end

    function [result, info]=MakeHeff(A, C, tol)
        if nargin == 2
            tol = 1e-14;
        end

        function result=P_NullSpace(x)
            % Projecting x on the nullspace of 1 - T

            x = reshape(x, bond, bond);
            result = trace(C' * x * C) * eye(bond);
        end

        function result=Transfer(x)
            % Doing (1 - (T - P)) @ x

            x = reshape(x, bond, bond);
            result = x(:);
            res = P_NullSpace(x);
            result = result + res(:);
            tmp = x * reshape(A, bond, []);
            res = reshape(A, [], bond)' * reshape(tmp, [], bond);
            result = result - res(:);
        end

        AA = reshape(A, [], bond) * reshape(A, bond, []);
        HAA = H_2site(AA);
        h = reshape(AA, [], bond)' * reshape(HAA, [], bond);

        temp = h - P_NullSpace(h);
        [r, info] = bicgstab(@Transfer, temp(:));

        r = reshape(r, bond, bond);
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
        AlHcc = reshape(Al(), [], bond) * reshape(Hcc, bond, []);
        error = Norm(HAcAc - 2 * AlHcc(:)) / (2 * abs(energy));
    end

    function [A, c, info]=MakeCanonical(AR, C_in, tol)
        if nargin == 2
            tol = 1e-14;
        end

        A = AR;
        c = eye(bond);

        diff = 1;
        iterations = 1;

        function result=Transfer(x)
            xA = reshape(x, bond, bond) * reshape(A, bond, []);
            result = reshape(A, [], bond)' * reshape(xA, [], bond);
            result = result(:);
        end

        while diff > tol
            iterations = iterations + 1;
            [w, d] = eigs(@Transfer, bond * bond);
            [u, s, v] = svd(reshape(w(:, 1), bond, bond));
            sqrt_eps = sqrt(eps(1.0));
            s = diag(s);
            s(:) = max(sqrt(s(:)), sqrt_eps);
            s = s / Norm(s);
            c1 = diag(s) * v';
            c1_inv = v * diag(s.^(-1));
            A = c1 * reshape(A, bond, []);
            A = reshape(A, [], bond) * c1_inv;
            A = (A / Norm(A)) * sqrt(bond);

            c = c1 * c;
            c = c / Norm(c);
            diff = Norm(reshape(w(:, 1), bond, bond) - eye(bond) * w(1));
        end
        c = c / Norm(c);
        info = [iterations, diff];
    end

    function info=set_uMPS(AC, C_in, canon, tol)
        if canon
            uar = poldec(reshape(AC, bond, []));
            ucr = poldec(reshape(C_in, bond, bond));
            ar = ucr' *  uar;
            [al, c, info] = MakeCanonical(Ar(), C_in, tol);
            ac = C() * reshape(Ar(), bond, []);
        else
            c = C_in;
            ua = poldec(reshape(AC, bond, []));
            uc = poldec(reshape(C, bond, bond));
            ar = ua * uc';
            al = uc' * ua;
            info = [0];
        end
    end

    function A=Ar()
        A = reshape(ar, bond, p, bond);
    end

    function A=Al()
        A = reshape(al, bond, p, bond);
    end

    function C=C()
        C = reshape(c, bond, bond);
    end

    function A=Ac()
        A = reshape(ac, bond, p, bond);
    end

    function print_info(i, energy, error, ctol, w1, w2, canon_info)
        % display(['Iter: ', i, ' E: ', energy, ' Error: ', error, 'tol: ', ctol, ' HAc: ', w1, ' Hc: ', w2, ' c_its: ', canon_info])
        display(['Iter: ', num2str(i), ' E: ', num2str(energy), ' Error: ', num2str(error)]);
    end

    H = four_site(HeisenbergInteraction(multi));
    p = size(H, 1);

    ac = randn(bond * p * bond, 1);
    c = randn(bond * bond, 1);
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
