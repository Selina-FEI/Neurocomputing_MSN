function [eval_result]  = evaluate_PL(X_test, y_test, X_train, w_global, W_local, hp)
%% FUNCTION evaluate_PL
%   Compute rmse, Rsquare, nmse, mae.
%   The values of rmse, nmse and mae are the lower the better,
%   while the value of Rsquare is the larger the better.
%
%% OUTPUT
%
%  rmse = ( rmse * N ) / N
%  mae = ( mae * N ) / N
%  Rsquare = rsquare / N
%  nmse = nmse / N
%
%  where
%     rmse = sqrt(sum((Y_pred - Y_true)^2)/ N)
%     mae = sum(abs(Y_pred - Y_true))/N_t
%     nmse = (sum((Y_pred - Y_true)^2)/ N_t)/sqrt(sum((Y_true).^2))
%     rsquare = 1 - sum((Y_true-Y_pred)^2)/sum((Y_true-mean(Y_true_all))^2)
%     Y_pred(n) = X{t}(:,n) * theta(:,n)
%     N     = length(Y)
%
%% INPUT
%   X_test: N x D
%   y_test: N x 1
%   X_train: N x D
%   w_global: D x 1 - weights of global model
%   W_local:  D x N - weights of local models
%   hp: hyperparameters

cla_flag = hp.cla_flag;
num_N = size(y_test,1);
calWeight = @(edgeDistance) exp(-(0.01).*edgeDistance);
findTheta = @(index) W_local(:,index);
if isempty(X_train)
    neighbors = hp.neighbors;
    distances = hp.distances;
else
    [neighbors, distances] =  knnsearch(X_train, X_test, 'K', hp.num_K, 'IncludeTies',true);
end
neighborTheta = cellfun(findTheta,neighbors,'UniformOutput',false);
neighborWeight = cellfun(calWeight,distances,'UniformOutput',false);
W_test_local  = cellfun(@weberSolver,neighborTheta, neighborWeight,'UniformOutput',false);
X_test_td = num2cell(X_test,2);
y_pred_local = sparse(blkdiag(X_test_td{:})) * cell2mat(W_test_local);
eval_result.yva_pred = y_pred_local;
if isempty(w_global)
    y_pred = y_pred_local;
else
    y_pred_global = X_test * w_global;
    y_pred = y_pred_local + y_pred_global;
end
if cla_flag
    y_pred = sigmoid(y_pred);
    auc = scoreAUC(y_test, y_pred);
    y_pred(y_pred>=0.5) = 1;
    y_pred(y_pred<0.5) = 0;
    acc = 1 - sum(~(y_pred == y_test)) / num_N;
    eval_result.acc = acc;
    eval_result.auc = auc;
    eval_result.metric_list = {'ACC', 'AUC'};
else
    mse = sum((y_pred-y_test).^2);
    mae = sum(abs(y_pred-y_test));
    rmse = sqrt(mse*num_N);
    nmse = mse / var(y_test) / num_N;
    ev = 1 - ( var(y_pred-y_test) / var(y_test) /num_N );
    eval_result.rmse = rmse / num_N;
    eval_result.mae = mae / num_N;
    eval_result.nmse = nmse;
    eval_result.ev = ev;
    eval_result.metric_list = {'RMSE', 'MAE', 'NMSE', 'EV'};
end

end
