function [w_global, W_local] = MSN_Lasso_boost(data, target, hp)
% *************************************************************************
% Multi-Level Sparse Network Lasso Booster (MSN_Lasso_boost)
% Accelerate the optimization of MSN Lasso by parallel computing
% Input:
%      Training data: data.X_train, target.y_train
%      Validation data: data.X_val, target.y_val
%      Hyperparameters: hp
% Output:
%      Global model parameters (D x 1): w_global
%      Local model parameters (D x N): W_local
%      Evaluation results: eval
% *************************************************************************

% 0. input information
cla_flag = hp.cla_flag;
X_train = data.X_train;
X_val   = data.X_val;
y_train = target.y_train;
y_val   = target.y_val;
[num_N, num_D] = size(X_train);
num_Nva = length(y_val);
clear data target

% 1. apply kmeans to seperate data
num_cluster = hp.num_cluster;
[idx, ~] = kmeans([X_train; X_val], num_cluster);
idx_tr = idx(1:num_N);
idx_va = idx(num_N+1:end);
idx_tr_cell = cell(num_cluster,1);
idx_va_cell = cell(num_cluster,1);
data_cell = cell(num_cluster,1);
target_cell = cell(num_cluster,1);
for k = 1 : num_cluster
    idx_tr_k = idx_tr==k;
    idx_va_k = idx_va==k;
    idx_tr_cell{k} = idx_tr_k;
    idx_va_cell{k} = idx_va_k;
    data_cell{k}.X_train = X_train(idx_tr_k,:);
    target_cell{k}.y_train = y_train(idx_tr_k);
    data_cell{k}.X_val = X_val(idx_va_k,:);
    target_cell{k}.y_val = y_val(idx_va_k,:);
end

% 2. conduct multiple MSN Lasso in parallel
W_cell = cell(num_cluster,1);
eval_cell = cell(num_cluster,1);
parfor k = 1 : num_cluster
    [W_cell{k}, eval_cell{k}] = MSN_Lasso(data_cell{k}, target_cell{k}, hp);
end
W_local = zeros(num_D, num_N);
yva_pred = zeros(num_Nva,1);
for k = 1 : num_cluster
    W_local(:,idx_tr_cell{k}) = W_cell{k};
    yva_pred(idx_va_cell{k}) = eval_cell{k}.yva_pred;
end
% eval = evaluate_booster(eval_cell, idx_va_cell, num_Nva, cla_flag);

% 3. train the global model by y_global = y_hat - y_local
Xtd_train = num2cell(X_train,2);
Xtd_train = sparse(blkdiag(Xtd_train{:}));
if cla_flag
    y_diff = y_train - sigmoid(Xtd_train * W_local(:));
    yva_diff = y_val - sigmoid(yva_pred);
else
    y_diff = y_train - Xtd_train * W_local(:);
    yva_diff = y_val - yva_pred;
end
w_global = initialize_w_ridge(X_train, y_diff, X_val, yva_diff, false);

end