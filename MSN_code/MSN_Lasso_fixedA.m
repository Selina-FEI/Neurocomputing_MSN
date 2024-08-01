function [W_train, eval] = MSN_Lasso_fixedA(data, target, hp)
% *************************************************************************
% Multi-Level Sparse Network Lasso (MSN_Lasso) for Personalized Learning
% Input:
%      Training data: data.X_train, target.y_train
%      Validation data: data.X_val, target.y_val
%      Hyperparameters: hp
% Output:
%      Local model parameters: W_train = a .* B
%      Evaluation results: eval
% *************************************************************************

% hyperparameters
cla_flag = hp.cla_flag;
init_flag = hp.init_flag;
warmup_flag = hp.warmup_flag;

% collect data
X_train = data.X_train;
X_val   = data.X_val;
y_train = target.y_train;
y_val   = target.y_val;
[num_N, num_D] = size(X_train);
clear data target

% initialization
if warmup_flag
    if init_flag
        [hp.neighbors, hp.distances] =  knnsearch(X_train, X_val, 'K', hp.num_K, 'IncludeTies',true);
        w = initialize_w_kmeans_ridge(X_train, y_train, X_val, y_val, hp);
    else
        w = initialize_w_ridge(X_train, y_train, X_val, y_val, cla_flag);
        w = kron(ones(num_N,1), w);
    end
else
    w = randn(num_N*num_D,1);
end

% construct the matrix D for structured sparsity
% D = construct_D(num_N, num_D, hp.mu);
D = construct_D_new(X_train, hp.mu, hp.num_K);
hp.num_G = length(D);

% main loop
X_train = num2cell(X_train,2);
X_train = sparse(blkdiag(X_train{:}));  % diagonalization for efficient computation
[w, obj] = update_w_PL(w, cell2mat(D), X_train, y_train, hp);
W_train = reshape(w,num_D,num_N);
eval.obj = obj;

end