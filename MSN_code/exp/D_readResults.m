% clc
clear all
% addpath(genpath('F:\HTEMTL_IJCAI23\demo_HTEMTL'))

select_alg = 'MSN-boost';  % 'MSN' or 'MSN-boost'
dataset = 'syn';

num_Folds = 10;
acc_list = zeros(num_Folds,1);
auc_list = zeros(num_Folds,1);
rmse_list = zeros(num_Folds,1);
mae_list  = zeros(num_Folds,1);
nmse_list = zeros(num_Folds,1);
ev_list   = zeros(num_Folds,1);
hp_list   = cell(num_Folds,1);
for fold_id = 1 : num_Folds
%     dataset_k = [dataset,'_',num2str(fold_id+10)];
    if fold_id == 10
        dataset_k = [dataset,'_',num2str(fold_id)];
    else
        dataset_k = [dataset,'_0',num2str(fold_id)];
    end
    log_name_k = ['D_log_',dataset_k,'_',select_alg,'.mat'];
    load(log_name_k)
    if hp.cla_flag
        acc_list(fold_id) = eval_test.acc;
        auc_list(fold_id) = eval_test.auc;
    else
        rmse_list(fold_id) = eval_test.rmse;
        mae_list(fold_id)  = eval_test.mae;
        nmse_list(fold_id) = eval_test.nmse;
        ev_list(fold_id)   = eval_test.ev;
    end
    hp_list{fold_id}  = hp;
end
acc_mean = mean(acc_list);
acc_std = std(acc_list) / sqrt(num_Folds);
auc_mean = mean(auc_list);
auc_std = std(auc_list) / sqrt(num_Folds);
disp(['error_mean=',num2str(acc_mean),', error_std=',num2str(acc_std)]);
disp(['auc_mean=',num2str(auc_mean),', auc_std=',num2str(auc_std)]);
log_name = ['D_finalRes_',dataset,'_',select_alg,'.mat'];
save(['results/', log_name], 'hp_list', 'acc_list', 'auc_list', 'acc_mean', 'acc_std', 'auc_mean', 'auc_std');
