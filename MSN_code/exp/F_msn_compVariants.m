%% val_and_store_TSCMTL: function description
function [outputs] = F_msn_compVariants(select_alg, dataset, cla_flag)

% dataset = 'syn';
% select_alg = 'MSN-boost';  % 'MSN' or 'MSN-boost'
% cla_flag = true;
% load(target_file);
if cla_flag
    num_Met = 2;
else
    num_Met = 4;
end
num_Folds = 10;


hp.cla_flag = cla_flag;       % true: classification, false: regression
hp.num_cluster = 4;        % number of data cluster for boosting
hp.mu      = 0.0;
hp.lambda1 = 1e-1;
hp.lambda2 = 1e-1;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 50;
hp.inner_iter_max = 100;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.num_K          = 5;         % for knn

value_set = [1, 2];
value_len = length(value_set);
comp_results = zeros(value_len, value_len, num_Met);
for k_id = 1 : value_len
    for p_id = 1 : value_len
        hp.k = value_set(k_id);
        hp.p = value_set(p_id);
        tmp_results = zeros(num_Met, num_Folds);
        for fold_id = 1 : num_Folds
            if fold_id == 10
                dataset_k = [dataset,'_',num2str(fold_id)];
            else
                dataset_k = [dataset,'_0',num2str(fold_id)];
            end
            load(dataset_k);
            [data] = preprocess_data(data);
            switch select_alg
                case 'PL'
                    hp.init_flag = true;     % true: kmeans+ridge; false: ridge only
                    [W_local, ~] = MSN_Lasso(data, target, hp);
                    eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, [], W_local, hp);
                case 'PL-boost'
                    hp.init_flag = false;     % true: kmeans+ridge; false: ridge only
                    [w_global, W_local] = MSN_Lasso_boost(data, target, hp);
                    eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, w_global, W_local, hp);
            end
            if cla_flag
                tmp_results(1, fold_id) = eval_test.acc;
                tmp_results(2, fold_id) = eval_test.auc;
            else
                tmp_results(1, fold_id) = eval_test.rmse;
                tmp_results(2, fold_id) = eval_test.mae;
                tmp_results(3, fold_id) = eval_test.nmse;
                tmp_results(4, fold_id) = eval_test.ev;
            end
        end
        comp_results(k_id, p_id, :) = mean(tmp_results, 2);
    end
end

save(['results/F_msn_compVariants_', dataset,'.mat'], 'hp', 'comp_results');

end