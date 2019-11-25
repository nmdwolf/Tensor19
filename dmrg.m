% verbosity: 0: Don't print anything.
%            1: Print results for the optimization.
%            2: Print intermediate result at every even chain length.
function E=dmrg(multi, bond, max_iter, verbosity)

    H = HeisenbergInteraction(multi);
    H = IsingInteraction(multi);
    p = size(H, 1);

    A = eye(1);
    chi = size(A, 1);
    L{1} = zeros(chi, p, chi, p);
    E = 0;
    Etot = 0;
    
    for i=1:max_iter
        [w,d] = eigs(@ApplyHeff, chi^2 * p^2);
        [A,trunc] = Truncate(w(:, 1), bond);
        sizes = size(A);
        chi = sizes(end);
        L{end+1} = UpdateHeff(A);
        
        E_ = (d(1) - Etot) / 2;
        dE = E - E_;
        E = E_;
        Etot = d(1);
        
        if verbosity >= 2
            disp(['Iter ', num2str(i), ' >>  M: ', num2str(chi), '  E:', num2str(E), ...
                  '  dE: ', num2str(dE), '  trunc:', num2str(trunc)]);
        end
    end

    if verbosity >= 1
        disp(['Final >> M: ', num2str(chi), '  E: ', num2str(E), ...
              '  dE: ', num2str(dE), '  trunc: ', num2str(trunc)]);
    end

    function [A,s]=Truncate(A, D)
        [u,s,v] = svd(reshape(A, chi * p, []));
        s = diag(s);
        cut = length(s);
        if cut > D
            cut = D;
        end
        u = u(:, 1:cut);
        
        A = reshape(u, chi, p, []);
        s = dot(s(cut+1:end),s(cut+1:end));
    end

    function x=ApplyHeff(A)
        A = reshape(A, chi * p, []);
        
        env_ = reshape(L{end}, chi * p, []);
        x = env_ * A + A * env_';
        x = reshape(x, chi, p, chi, p);
        
        A = permute(reshape(A, chi, p, chi, p), [1 3 2 4]);
        h = reshape(A, chi*chi, []) * permute(reshape(H, p*p, p*p), [2 1]);
        x = x + permute(reshape(h, chi, chi, p, p), [1 3 2 4]);

        x = x(:);
    end

    function env=UpdateHeff(A)
        Mp = sizes(1) * sizes(2);
        A_ = reshape(A, Mp, []);
        
        env = A_' * reshape(L{end}, Mp, Mp) * A_;
        env = kron(eye(p), env);
        env = reshape(env, chi, p, chi, p);
        
        A = reshape(A, [], p*chi);
        B = reshape(A' * A, p, chi, p, chi);
        B = reshape(permute(B, [2 4 1 3]), chi*chi, p*p);
        h = reshape(permute(H, [1 3 2 4]), p*p, p*p);
        env = env + permute(reshape(B * h, chi, chi, p, p), [1 3 2 4]);
    end

    function [z,pl,mn]=S_operators(multi)
    
        if nargin==0
            j = 1/2;
            multi = 2;
        else
            j = (multi - 1) / 2;
        end
        m = (0:multi-1) - j;

        z = diag(m);
        pl = zeros(length(m));
        temp = sqrt((j - m) .* (j + m + 1));
        pl(2:(multi+1):end) = temp(1:end-1);
        
        mn = pl.';
    end

    function H=HeisenbergInteraction(multi)
    
        if nargin==0
            multi = 2;
        end
        [Sz, Sp, Sm] = S_operators(multi);
        
        H = 0.5 * (kron(Sp, Sm) + kron(Sm, Sp)) + kron(Sz, Sz);
        H = reshape(H,multi,multi,multi,multi); 
    end

    function H=IsingInteraction(multi, J)
        % Returns Ising interaction between two sites.
        %
        % This is given by:
        %    1/2 * (S_1^+  + S_1^- + S_2^+ + S_2^-) + J * S_1^z S_2^z
        %
        % Interaction is given in a dense matrix:
        %    Σ H_{1', 2', 1, 6} |1'〉|2'〉〈1|〈2|
        
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
end