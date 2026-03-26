function [y_soft, w, y_k, H_k] = multiscale_entropy_weighted_labels(X, Y, K)
% =========================================================================
% Soft Label Fusion Mechanism Based on Multi-scale Neighborhood and Entropy
% Input:
%   X      : n × d feature matrix
%   Y      : n × L multi-label matrix, values in {0,1} or {-1,1}
%   K      : Number of scales (granularity levels)
% Output:
%   y_soft : n × L final fused soft label probability matrix
%   w      : 1 × K adaptive weight for each scale
%   y_k    : n × L × K initial soft label matrix for each scale
%   H_k    : 1 × K "average neighborhood information entropy" for each scale
% =========================================================================

[n, d] = size(X);
L = size(Y, 2);
eps_val = 1e-12;

%% 1. Calculate base scale parameter (Theta)
theta = mean(std(X, 0, 1));

%% 2. Construct Euclidean distance matrix
D = zeros(n, n);
for i = 1:n
    for j = 1:n
        D(i,j) = norm(X(i,:) - X(j,:)) / sqrt(d);
    end
end

%% 3. Multi-scale soft label estimation & neighborhood entropy calculation
y_k  = zeros(n, L, K);
H_k  = zeros(1, K);      % Accumulated entropy for each scale
cnt_k = zeros(1, K);     % Counter for averaging later

for k = 0:K-1
    e = -(k + 1 - K + floor(K/2));
    theta_k = theta * 2^e; % Dynamically adjust the neighborhood sphere radius

    for i = 1:n
        idx = find(D(i,:) <= theta_k);

        % Extreme case: Only the instance itself is in the neighborhood (lack of local info)
        if length(idx) == 1
            y_k(i,:,k+1) = 0.75 * Y(i,:) - 0.25;
            H_k(k+1) = H_k(k+1) + L * 1; % Assign base penalty entropy
            cnt_k(k+1) = cnt_k(k+1) + L;
            continue;
        end

        for l = 1:L
            if Y(i,l) == -1
                y_k(i,l,k+1) = -1;
            else
                % Count the proportion of positive/negative samples in the local neighborhood
                pos = sum(Y(idx,l) == 1);
                neg = sum(Y(idx,l) == 0 | Y(idx,l) == -1);
                y_k(i,l,k+1) = 0.5 + 0.5 * (pos - neg) / length(idx);
            end

            % Calculate Bernoulli Entropy within the neighborhood sphere
            pos = sum(Y(idx,l) == 1);
            p = pos / length(idx);

            H_il = - p * log(p + eps_val) - (1 - p) * log(1 - p + eps_val);

            H_k(k+1) = H_k(k+1) + H_il;
            cnt_k(k+1) = cnt_k(k+1) + 1;
        end
    end
end

%% 4. Entropy-driven scale weight allocation
H_k = H_k ./ cnt_k; % Calculate average information entropy for each scale
w = exp(-H_k);      % The smaller the entropy (higher purity), the larger the weight
w = w / sum(w);     % Normalize weights

%% 5. Multi-scale soft label fusion (Weighted sum)
y_soft = zeros(n, L);
for k = 1:K
    y_soft = y_soft + w(k) * y_k(:,:,k);
end

end