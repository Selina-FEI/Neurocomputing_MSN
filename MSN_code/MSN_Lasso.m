function [W_train, eval] = MSN_Lasso(data, target, hp)
% *************************************************************************
% Multi-Level Sparse Network Lasso (MSN_Lasso) for Personalized Learning
% Basic Optimization
% Input:
%      Training data: data.X_train, target.y_train
%      Validation data: data.X_val, target.y_val
%      Hyperparameters: hp
% Output:
%      Local model parameters: W_train = a .* B
%      Evaluation results: eval
% *************************************************************************

% hyperparameters
p = hp.p;
k = hp.k;
lambda1 = hp.lambda1;
lambda2 = hp.lambda2;
cla_flag = hp.cla_flag;
init_flag = hp.init_flag;
warmup_flag = hp.warmup_flag;
absTol = hp.absTol;
outer_iter_max = hp.outer_iter_max;

% collect data
X_train = data.X_train;
X_val   = data.X_val;
y_train = target.y_train;
y_val   = target.y_val;
[num_N, num_D] = size(X_train);
clear data target

% initialization
[hp.neighbors, hp.distances] =  knnsearch(X_train, X_val, 'K', hp.num_K, 'IncludeTies',true);
if warmup_flag
    if init_flag
        w = initialize_w_kmeans_ridge(X_train, y_train, X_val, y_val, hp);
%         hp = rmfield(hp, {'neighbors','distances'});
    else
        w = initialize_w_ridge(X_train, y_train, X_val, y_val, cla_flag);
        w = kron(ones(num_N,1), w);
    end
else
    w = randn(num_N*num_D,1);
end

% construct the matrix D for structured sparsity
D = construct_D(X_train, hp.mu, hp.num_K, hp.flag_graph);
num_G = length(D);
Dt = D;
a  = ones(num_G,1);
hp.num_G = num_G;

% main loop
X_train = num2cell(X_train,2);
X_train = sparse(blkdiag(X_train{:}));  % diagonalization for efficient computation
obj = [];
obj = [obj; obj_value_MSN(X_train, y_train, w, a, Dt, hp)];    
for iter = 2 : outer_iter_max
    % 1. update a (TBD: randomly update?)
    for g = 1 : num_G
        a(g) = nthroot((lambda2/lambda1) * norm(D{g}*w, p).^p + eps, k+p);
        Dt{g} = D{g} ./ a(g);
    end

    % 2. update w
    [w, ~] = update_w_PL(w, cell2mat(Dt), X_train, y_train, hp);  

    % 3. convegence analysis
    obj = [obj; obj_value_MSN(X_train, y_train, w, a, Dt, hp)];
    if obj_converges(obj,absTol) || iter == outer_iter_max
        W_train = reshape(w,num_D,num_N);
        eval = evaluate_PL(X_val, y_val, [], [], W_train, hp);
        eval.obj = obj;
        break;
    end
end
end

% objective value
function [obj] = obj_value_MSN(X_train, y_train, w, a, Dt, hp)
obj = sum_loss_STL(X_train, y_train, w, hp.cla_flag) + ...
    hp.lambda1 * norm(a,hp.k).^hp.k + ...
    hp.lambda2 * norm(cell2mat(Dt)*w,hp.p).^hp.p;
end