%% val_and_store_TSCMTL: function description
function [outputs] = D_val_and_store_MSN_cla(target_file, log_name)

select_alg = 'MSN-boost';  % 'MSN' or 'MSN-boost'
hp.cla_flag = true;
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
%                 if cla_flag && hp.lambda > hp.gamma
%                     recorder{i}{j}{k}{p} = 1;
%                     continue;
%                 end
                fprintf('(%d/%d): lambda1: %f, lambda2: %f, mu: %f phi: %f num_cluster: %f',  hp_index, length(lambda1_list) * ...
                    length(lambda2_list) * length(mu_list) * length(cluster_list), hp.lambda1, hp.lambda2, hp.mu, hp.num_cluster);
                [w_global, W_local] = MSN_Lasso_boost(data, target, hp);
                eval_val = evaluate_PL(data.X_val, target.y_val, data.X_train, w_global, W_local, hp);
                disp(["Validation: ACC=",num2str(eval_val.acc),", AUC=",num2str(eval_val.auc)]);
                recorder{i}{j}{k}{p} = eval_val.acc;
                hp_index = hp_index + 1;
            end
        end
    end
end

% find out the optimal hyper parameters
min_acc_val = 0;
opt_lambda1 = 0;
opt_lambda2 = 0;
opt_mu      = 0;
opt_num_cluster = 0;
for i = 1:length(lambda1_list)
    for j = 1:length(lambda2_list)
        for k = 1:length(mu_list)
            for p = 1:length(cluster_list)
                curr_acc = recorder{i}{j}{k}{p};
                if curr_acc > min_acc_val
                    min_acc_val = curr_acc;
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