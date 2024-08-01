% ER
clc
clear all

% AWA_data1_wiL = 0.139670588;
% AWA_data1_noL = 0.143964706;
% AWA_data2_wiL = 0.1316;
% AWA_data2_noL = 0.1340;

cla_flag = true;
data_set = {'mnist10', 'cifar', 'awa50'};
alg_set = {'MSN', 'MSN-a'};
alg_len = length(alg_set);
data_len = length(data_set);

if cla_flag
    num_Met = 2;
    met_set = {'ACC', 'AUC'};
else
    num_Met = 4;
    met_set = {'RMSE', 'MAE', 'NMSE', 'EV'};
end

hp.k       = 2;
hp.p       = 2;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 50;
hp.inner_iter_max = 100;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.init_flag      = true;      % true: kmeans+ridge; false: ridge only
hp.num_K          = 5;         % for knn
hp.cla_flag = cla_flag;       % true: classification, false: regression

comp_results = zeros(alg_len, num_Met, num_Folds);
for data_id = 1 : length(data_set)
    dataset = data_set{data_id};
%     comp_results = zeros(alg_len, num_Met, num_Folds);
    for alg_id = 1 : alg_len
        this_alg = alg_set{alg_id};
        tmp_results = zeros(num_Met, num_Folds);
        for fold_id = 1 : 10
            if fold_id == 10
                dataset_k = [dataset,'_',num2str(fold_id)];
            else
                dataset_k = [dataset,'_0',num2str(fold_id)];
            end
            [data] = preprocess_data(data);
            switch this_alg
                case 'MSN'
                    [W_local, ~] = MSN_Lasso(data, target, hp);
                case 'MSN-a'
                    [W_local, ~] = MSN_Lasso_fixedA(data, target, hp);
            end
            eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, [], W_local, hp);
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
        comp_results(data_id, alg_id, :) = mean(tmp_results, 2);
    end
end
save(['results/G_msn_multiLevel.mat'], 'hp', 'comp_results');

for met_id = 1 : num_Met
    b = bar(comp_resutls(:,:,met_id));
%     b = bar([AWA_data1_noL, AWA_data1_wiL; ...
%         AWA_data2_noL, AWA_data2_wiL]);
%     ylim([0.125 0.145])
%     b(2).FaceColor = [0.25 0.41 0.88];
%     b(1).FaceColor = [0.25 0.88 0.82];

    set(gca,'xticklabel', data_set, 'FontSize', 22,'fontweight','bold' )
    ylabel(met_set{met_id}, 'FontSize', 22, 'fontweight','bold')
    legend('Fixed \alpha','Update \alpha', 'FontSize', 22);
    % legend('Fixed \alpha','Update \alpha', 'FontSize', 22, 'fontweight','bold', 'interpreter', 'latex');
    saveas(fig,['G_msn_multiLevel_',met_set{met_id},'.png']);
end
