%% single_task_learn_reg: function description
function [w_init] = initialize_w_kmeans_ridge(X_train, y_train, X_val, y_val, hp)

cluster_list = [4, 6, 8, 10]; 
lambda_list  = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1];
rmse = Inf;
iter_max = 1000; % maximal iteration numbers
alpha = 1e+1; % initial step size
EPS_G = 1e-6;
acc = 0;
cla_flag = hp.cla_flag;
lambda_len = length(lambda_list);
cluster_len = length(cluster_list);
[num_N, num_D] = size(X_train);
for cluster_id = 1 : cluster_len
    num_cluster = cluster_list(cluster_id);
    [idx, ~] = kmeans(X_train, num_cluster);
    for lambda_id = 1 : lambda_len
        lambda = lambda_list(lambda_id);
        W = zeros(num_D,num_N);
        for k = 1 : num_cluster
            idx_k = idx==k;
            X_k = X_train(idx_k,:);
            w_k = base_learner(X_k, y_train(idx_k), lambda, cla_flag);
            W(:, idx_k) = w_k * ones(1,nnz(idx_k));
        end
        eval_res = evaluate_PL(X_val, y_val, [], [], W, hp);
        if cla_flag
            if eval_res.acc >= acc
                acc = eval_res.acc;
                auc = eval_res.auc;
                w_init = W(:);
            end
        else
            if eval_res.rmse <= rmse
                rmse = eval_res.rmse;
                mae  = eval_res.mae;
                w_init = W(:);
            end
        end
    end
end
% if cla_flag
%     disp(["ACC=",num2str(acc),", MAE=",num2str(auc)]);
% else
%     disp(["RMSE=",num2str(rmse),", MAE=",num2str(mae)]);
% end
end