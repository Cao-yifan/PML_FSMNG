function [W, bt] = PML_FSMNG(X_train, Y_train, alpha, beta, gamma)
% =========================================================================
% Feature Selection Optimizer based on L2,1-Norm and Graph Regularization
% Input:
%   X_train : n x d feature matrix (will be transposed internally)
%   Y_train : n x c label matrix (soft labels)
%   alpha   : Weight for the L2,1-norm penalty term on W
%   beta   : Weight for the graph regularization term
%   gamma    : Weight for the entropy term (mapped internally)

% Output:
%   W       : d x c feature weight matrix
%   bt      : c x 1 bias vector
% =========================================================================
%   p       : Norm parameter (typically 1 for L2,1-norm)
p = 1;
rng(43); % Fix random seed for reproducibility
[dim, n] = size(X_train);
label_num = size(Y_train, 2);

% Initialize variables
W = rand(dim, label_num);
bt = rand(label_num, 1);
V = rand(n, n);
L_v = compute_laplacian_from_V(V);

% Initialize diagonal matrices
T = X_train' * W + ones(n, 1) * bt' - Y_train;
G0 = diag(0.5 ./ sqrt(sum(T.*T, 2) + eps));
G2 = eye(dim);

m = sum(sum(G0)) + beta * sum(sum(L_v));
H = eye(n) - (1/m) * ones(n,n) * G0;
N = eye(n) - (1/m) * ones(n,n) * (G0 + beta * L_v);

iter = 1;
obji = 1;

% Alternating optimization loop
while 1
    % 1. Update weight matrix W
    term1 = X_train * N' * (G0 + beta * L_v) * N * X_train' + alpha * G2;
    term2 = X_train * N' * (G0 * H + beta * L_v *(H - 1)) * Y_train;
    W = pinv(term1) * term2;

    % 2. Update diagonal matrix G2 (Derivative of L2,1 norm)
    wc = (sum(W .* W, 2) + eps).^(1 - p/2);
    Gw = 1 ./ ((2/p) * wc);
    G2 = spdiags(Gw, 0, dim, dim);

    % 3. Update auxiliary variable T and G0
    T = X_train' * W + ones(n, 1) * bt' - Y_train;
    G0 = diag(0.5 ./ sqrt(sum(T.*T, 2) + eps));

    % 4. Update H, N, and bias bt
    m = sum(sum(G0)) + beta * sum(sum(L_v));
    H = eye(n) - (1/m) * ones(n,n) * G0;
    N = eye(n) - (1/m) * ones(n,n) * (G0 + beta * L_v);
    bt = (1/m) * Y_train' * G0 * ones(n, 1) - (1/m) * W' * X_train * (G0 + beta * L_v') * ones(n, 1);

    % 5. Update affinity matrix V and Laplacian matrix L_v
    V = compute_affinity_V(X_train', W, gamma/beta);
    L_v = compute_laplacian_from_V(V);

    % Check convergence
    objective(iter) = compute_objective(X_train, Y_train, W, bt, L_v, V, beta, gamma, alpha);
    cver = abs((objective(iter) - obji)/obji);

    if mod(iter, 100) == 0
        fprintf('Iteration %d: Objective = %.12f\n', iter, obji);
    end

    obji = objective(iter);
    iter = iter + 1;

    if (cver < 1e-5 && iter > 2) || (iter == 21)
        break;
    end
end
end

%% ================= Sub-functions ================= %%

function V = compute_affinity_V(X, W, gamma)
% Compute row-normalized affinity matrix (Softmax mechanism)
    [n, ~] = size(X);
    U = X * W;
    V = zeros(n, n);
    for a = 1:n
        for b = 1:n
            diff = U(a,:) - U(b,:);
            V(a,b) = exp(-(diff * diff') / (2 * gamma));
        end
        % Row normalization
        V(a,:) = V(a,:) / sum(V(a,:));
    end
end

function L_v = compute_laplacian_from_V(V)
% Compute graph Laplacian matrix
    www = (V + V') / 2;
    d = sum(www, 2);
    E = diag(d);
    L_v = E - www;
end

function obj_val = compute_objective(X_train, Y_train, W, bt, L_v, V, beta, gamma, alpha)
% Compute the overall objective function value
    X_train = X_train';
    n = size(X_train, 1);

    F = X_train * W + ones(n, 1) * bt';
    R = F - Y_train;

    l21_norm = sum(sqrt(sum(R.^2, 2)));
    graph_reg = trace(F' * L_v * F);

    eps_val = 1e-12;
    entropy_term = sum(sum(V .* log(V + eps_val)));

    W_l21 = sum(sqrt(sum(W.^2, 2)));

    % Combine terms into the final objective value
    obj_val = l21_norm + beta * graph_reg + gamma * entropy_term + alpha * W_l21;
end