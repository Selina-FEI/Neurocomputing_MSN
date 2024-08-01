%% val_and_store_TSCMTL: function description
function [outputs] = D_val_and_store_MSN_reg(target_file, log_name)

select_alg = 'MSN-boost';  % 'MSN' or 'MSN-boost'
hp.cla_flag = false;
load(target_file);

[data] = preprocess_data(data);

% awa50
lambda1_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1];
lambda2_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1];
mu_list      = 0 : 0.2 : 1;
cluster_list = 5 : 5 : 20;

hp_index = 1;
for i = 1:length(lambda1_list)
    for j = 1:length(lambda2_list)
        for k = 1:length(mu_list)
            for p = 1:length(cluster_list)
                hp.lambda1 = lambda1_list(i);
                hp.lambda2 = lambda2_list(j);
                hp.mu = mu_list(k);
                hp.num_cluster = cluster_list(p);
                fprintf('(%d/%d): lambda1: %f, lambda2: %f, mu: %f phi: %f num_cluster: %f',  hp_index, length(lambda1_list) * ...
                    length(lambda2_list) * length(mu_list) * length(cluster_list), hp.lambda1, hp.lambda2, hp.mu, hp.num_cluster);
                [w_global, W_local] = MSN_Lasso_boost(data, target, hp);
                eval_val = evaluate_PL(data.X_val, target.y_val, data.X_train, w_global, W_local, hp);
                disp(["Validation: RMSE=",num2str(eval_val.rmse),", MAE=",num2str(eval_val.mae)]);
                recorder{i}{j}{k}{p} = eval_val.rmse;
                hp_index = hp_index + 1;
            end
        end
    end
end

% find out the optimal hyper parameters
min_rmse_val = Inf;
opt_lambda1 = 0;
opt_lambda2 = 0;
opt_mu      = 0;
opt_num_cluster = 0;
for i = 1:length(lambda1_list)
    for j = 1:length(lambda2_list)
        for k = 1:length(mu_list)
            for p = 1:length(cluster_list)
                curr_rmse = recorder{i}{j}{k}{p};
                if curr_rmse < min_rmse_val
                    min_rmse_val = curr_rmse;
                    opt_lambda1 = lambda1_list(i);
                    opt_lambda2 = lambda2_list(j);
                    opt_mu      = mu_list(k);
                    opt_num_cluster = cluster_list(p);
                end
            end
        end
    end
end

% train on the full training set and test on the test set
hp.lambda1 = opt_lambda1;
hp.lambda2 = opt_lambda2;
hp.mu      = opt_mu;
hp.num_cluster = opt_num_cluster;
[w_global, W_local] = MSN_Lasso_boost(data, target, hp);
eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, w_global, W_local, hp);
save(['results/', log_name], 'hp', 'w_global', 'W_local', 'eval_test','recorder');

end