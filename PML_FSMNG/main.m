% =========================================================================
% Main Script for Multi-label Feature Selection and Model Evaluation
% Description: Entry point of the project. Run directly to execute
%              the entire algorithm pipeline.
% Default dataset: 'music_emotion.mat' is used as a demonstration.
% =========================================================================

clear; clc;

%% 1. Environment Initialization & Data Loading
filename = 'music_emotion.mat'; % Modify the dataset name here if needed
load(filename);
fprintf('Successfully loaded dataset: %s\n', filename);

%% 2. Data Preprocessing
% Transpose label matrix to fit algorithm dimensions (N x L)
candidate_labels = candidate_labels';
target = target';

% Convert 0s in true labels to -1
target(target == 0) = -1;

% Filter out invalid samples with all -1s (i.e., no valid positive labels)
idx = any(target ~= -1, 2);
X = data(idx, :);
y = candidate_labels(idx, :);
y_true = target(idx, :);
y(y == 0) = -1; % Convert 0s in candidate labels to -1 as well

% Min-Max (0,1) normalization for features
min_val = min(X, [], 1);
max_val = max(X, [], 1);
range_val = max_val - min_val;
range_val(range_val == 0) = 1;  % Avoid division by zero
X = (X - min_val) ./ range_val;
fprintf('Feature Min-Max normalization completed.\n');

%% 3. Cross-Validation Setup
% Convert multi-label vectors to string format (e.g., "1_-1_1") for stratified sampling
y_str = strings(size(y, 1), 1);
for i = 1:size(y, 1)
    y_str(i) = strjoin(string(double(y(i, :))), '_');
end

% 5-fold stratified cross-validation setup
rng(43);
warning('off', 'stats:cvpartition:KFoldMissingGrp');
cv = cvpartition(y_str, 'KFold', 5);

%% 4. Hyperparameter Grid Setup
alpha_list = [1];
beta_list = [1e-2];
gamma_list = [100];

% Calculate total number of tasks
[num_alpha, num_beta, num_gamma] = ...
    deal(length(alpha_list), length(beta_list), length(gamma_list));

total_jobs = num_alpha * num_beta * num_gamma;

% Containers for storing results
local_results = cell(total_jobs, 1);
local_configs = cell(total_jobs, 1);
percent = 0; % Partial label missing rate setting

%% 5. Core Experiment Main Loop
for para_idx = 1:total_jobs
    % Map 1D index to multi-dimensional parameter indices
    [i, j, m] = ind2sub([num_alpha, num_beta, num_gamma], para_idx);

    % Get current parameter combination
    alpha = alpha_list(i);
    beta = beta_list(j) ; % Note the dependency here
    gamma = gamma_list(m)*beta;
    fprintf('\n[%d / %d] Evaluating Parameters: alpha=%.4f, beta=%.4f, gamma=%.4f\n', ...
            para_idx, total_jobs, alpha, beta, gamma);

    % Initialize storage array for 5-fold evaluation metrics (12 metrics)
    metrics = zeros(5, 12);
    selected_features_all = cell(5, 1);

    for k = 1:5
        % Extract training and testing sets for the current fold
        X_train = X(cv.training(k), :);
        X_test  = X(cv.test(k), :);
        y_train = y(cv.training(k), :);
        y_test  = y(cv.test(k), :);

        y_train_true = y_true(cv.training(k), :);
        y_test_true  = y_true(cv.test(k), :);
        l_labels = size(y_train, 2);

        % Get partial labels
        [PartialLabel, ~] = getPartialLabel(y_train, percent, 1); % bQuiet = 1 (Silent mode)

        t_start = tic;

        % --- Core Algorithm: Multi-scale entropy weighting and feature selection ---
        [y_soft, ~, ~, ~] = multiscale_entropy_weighted_labels(X_train, PartialLabel, 3);
        [W, ~] = PML_FSMNG(X_train', y_soft, alpha, beta, gamma);

        elapsed_time = toc(t_start);
        fprintf('  - Fold %d feature selection time: %.4f seconds\n', k, elapsed_time);

        % Extract top 20% important features based on weights
        row_norms = sqrt(sum(W.^2, 2));
        [~, idx_sort] = sort(row_norms, 'descend');
        n_selected = round(0.2 * size(X, 2));
        selected_features = idx_sort(1:n_selected);
        selected_features_all{k} = selected_features;

        % --- Model Evaluation: MLKNN ---
        [Prior, PriorN, Cond, CondN] = MLKNN_train(X_train(:, selected_features), y_train_true', 10, 1);

        [HL, RL, OE, CV, AP, MA, MI, AC, ~, ~] = ...
            MLKNN_test(X_train(:, selected_features), y_train_true', ...
                       X_test(:, selected_features), y_test_true', ...
                       10, Prior, PriorN, Cond, CondN);

        % Store metrics [HL, RL, OE, CV, AP, MA, MI, AC, RECO, posRECO, negRECO, TIME]
        metrics(k, :) = [HL, RL, OE, CV, AP, MA, MI, AC, 1, 1, 1, elapsed_time];
    end

    %% 6. Summarize and Save Evaluation Results for Current Parameter Config
    avg_m = mean(metrics);
    std_m = std(metrics);

    avg_CV = avg_m(4) / l_labels;
    std_CV = std_m(4) / l_labels;

    % Save metrics as a struct
    local_results{para_idx} = struct('HammingLoss', avg_m(1), 'RankingLoss', avg_m(2), 'OneError', avg_m(3), ...
        'Coverage', avg_CV, 'Average_Precision', avg_m(5), 'macrof1', avg_m(6), 'microf1', avg_m(7), 'SubsetAccuracy', avg_m(8), 'RecoveryAccuracy', avg_m(9));

    % Save configuration as a struct
    local_configs{para_idx} = struct('alpha', alpha, 'beta', beta, 'gamma', gamma, 'selected_features', {selected_features_all}, ...
        'HammingLoss', sprintf('%.4f ± %.4f', avg_m(1), std_m(1)), ...
        'RankingLoss', sprintf('%.4f ± %.4f', avg_m(2), std_m(2)), ...
        'OneError', sprintf('%.4f ± %.4f', avg_m(3), std_m(3)), ...
        'Coverage', sprintf('%.4f ± %.4f', avg_CV, std_CV), ...
        'Average_Precision', sprintf('%.4f ± %.4f', avg_m(5), std_m(5)), ...
        'macrof1', sprintf('%.4f ± %.4f', avg_m(6), std_m(6)), ...
        'microf1', sprintf('%.4f ± %.4f', avg_m(7), std_m(7)), ...
        'SubsetAccuracy', sprintf('%.4f ± %.4f', avg_m(8), std_m(8)));

end

%% 7. Print Global Optimal Results Summary
fprintf('\n================ Global Optimal Parameters & Metrics ================\n');
metric_names = {'HammingLoss', 'RankingLoss', 'OneError', 'Coverage', ...
                'Average_Precision', 'macrof1', 'microf1', 'SubsetAccuracy'};

for m = 1:numel(metric_names)
    metric = metric_names{m};
    values = zeros(total_jobs, 1);
    for idx = 1:total_jobs
        values(idx) = local_results{idx}.(metric);
    end

    % Determine whether to maximize or minimize (Maximize AP, F1; Minimize HL, RL, etc.)
    if ismember(metric, {'Average_Precision', 'macrof1', 'microf1', 'SubsetAccuracy'})
        [~, best_idx] = max(values);
        type_str = 'Max';
    else
        [~, best_idx] = min(values);
        type_str = 'Min';
    end

    best_cfg = local_configs{best_idx};

    fprintf('\n>> Optimal Config for [%s %s] <<\n', metric, type_str);
    fprintf('Parameters: alpha=%.4f, beta=%.4f, gamma=%.4f\n', ...
            best_cfg.alpha, best_cfg.beta, best_cfg.gamma);
    fprintf('------------------------------------------------------\n');
    fprintf('HammingLoss       : %s\n', best_cfg.HammingLoss);
    fprintf('RankingLoss       : %s\n', best_cfg.RankingLoss);
    fprintf('OneError          : %s\n', best_cfg.OneError);
    fprintf('Coverage          : %s\n', best_cfg.Coverage);
    fprintf('Average_Precision : %s\n', best_cfg.Average_Precision);
    fprintf('Macro-F1          : %s\n', best_cfg.macrof1);
    fprintf('Micro-F1          : %s\n', best_cfg.microf1);
    fprintf('SubsetAccuracy    : %s\n', best_cfg.SubsetAccuracy);
    fprintf('======================================================\n');
end