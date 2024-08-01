% This is a demo program of comparing methods on the Synthetic data

clc
clear
close all
rng('default')
addpath('results')
addpath(genpath('C:\Users\DELL\Desktop\MSN_Lasso'))

%% Global parameter settting
cla_flag = false; 
this_var = 'num_C';
switch cla_flag
    case true
        met_len  = 3;  % time, acc, auc
    case false
        met_len  = 5;  % time, rmse, mae, rsquare, nmse
end


%% Experimental setting
% scale_set = 200:100:1000;
cluster_set = 2:2:10;
num_Ntr_syn = 200;


%% Exp size
cluster_len  = length(cluster_set);

% hyperparameters
hp.cla_flag = false;       % true: classification, false: regression
% hp.num_cluster = 4;        % number of data cluster for boosting
hp.k       = 2;
hp.p       = 2;
hp.mu      = 0.0;
hp.rho     = 1e+0;
hp.lambda1 = 1e-1;
hp.lambda2 = 1e-1;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 50;
hp.inner_iter_max = 100;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.num_K          = 5;         % for knn
hp.init_flag = false;    % true: kmeans+ridge; false: ridge only

% %% Generate synthetic datasets
% for scale_id = 1 : scale_len
%     this_numN = scale_set(scale_id);
%     [data, target, model] = generate_syndata_PL(this_numN, cla_flag);
%     [data] = preprocess_data(data);
%     current_info = ['syndata_',num2str(this_numN)];
%     save(['results/E_',current_info,'.mat'],'data','target','model');
% end

%% Evaluation of preformance
eval_results = zeros(cluster_len, met_len);
[data, target, model] = generate_syndata_PL(num_Ntr_syn, cla_flag);
[data] = preprocess_data(data);
for cluster_id = 1 : cluster_len
    this_num_C = cluster_set(cluster_id);  
    hp.num_cluster = this_num_C;        % number of data cluster for boosting
    tic;
    [w_global, W_local, eval] = MSN_Lasso_boost(data, target, hp);
    eval_results(cluster_id, 1) = toc;
    eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, w_global, W_local, hp);
    switch cla_flag
        case true
            eval_results(cluster_id, 2) = eval_test.acc;
            eval_results(cluster_id, 3) = eval_test.auc;
        case false
            eval_results(cluster_id, 2) = eval_test.rmse;
            eval_results(cluster_id, 3) = eval_test.mae;
            eval_results(cluster_id, 4) = eval_test.nmse;
            eval_results(cluster_id, 5) = eval_test.ev;
    end
end
save(['results/C_msn_varyNumCluster_',num2str(num_Ntr_syn),'.mat'],'eval_results','hp');

