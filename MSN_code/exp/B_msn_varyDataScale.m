% This is a demo program of comparing methods on the Synthetic data

clc
clear
close all
rng('default')
addpath('results')
addpath(genpath('C:\Users\DELL\Desktop\MSN_Lasso'))

%% Global parameter settting
cla_flag = false; 
this_var = 'numN';
switch cla_flag
    case true
        met_len  = 3;  % time, acc, auc
    case false
        met_len  = 5;  % time, rmse, mae, rsquare, nmse
end


%% Experimental setting
alg_set = {'PL','PL-boost'};
scale_set = 100:100:1000;
% scale_set = 20:10:50;


%% Exp size
alg_len  = length(alg_set);
scale_len = length(scale_set);

% hyperparameters
hp.cla_flag = false;       % true: classification, false: regression
hp.num_cluster = 4;        % number of data cluster for boosting
hp.k       = 2;
hp.p       = 2;
hp.mu      = 0.2;
hp.rho     = 1e+0;
hp.lambda1 = 1e-1;
hp.lambda2 = 1e-1;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 200;
hp.inner_iter_max = 200;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.num_K          = 5;         % for knn

% %% Generate synthetic datasets
% for scale_id = 1 : scale_len
%     this_numN = scale_set(scale_id);
%     [data, target, model] = generate_syndata_PL(this_numN, cla_flag);
%     [data] = preprocess_data(data);
%     current_info = ['syndata_',num2str(this_numN)];
%     save(['results/E_',current_info,'.mat'],'data','target','model');
% end

%% Evaluation of preformance
eval_results = zeros(scale_len, alg_len, met_len);
for scale_id = 1 : scale_len
%     rng(scale_id)
    this_numN = scale_set(scale_id);
    [data, target, model] = generate_syndata_PL(this_numN, cla_flag);
    [data] = preprocess_data(data);
    for alg_id = 1 : alg_len
        this_alg = alg_set{alg_id};
        current_info = [this_var,'_',num2str(this_numN),'_',this_alg];
        disp(current_info);
        tic;
        switch this_alg
            case 'PL'
                hp.init_flag = true;     % true: kmeans+ridge; false: ridge only
                [W_local, eval] = MSN_Lasso(data, target, hp);
            case 'PL-boost'
                hp.init_flag = false;    % true: kmeans+ridge; false: ridge only
                [w_global, W_local] = MSN_Lasso_boost(data, target, hp);
        end
        eval_results(scale_id, alg_id, 1) = toc;
        switch this_alg
            case 'PL'
                eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, [], W_local, hp);
            case 'PL-boost'
                eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, w_global, W_local, hp);
        end
        switch cla_flag
            case true
                eval_results(scale_id, alg_id, 2) = eval_test.acc;
                eval_results(scale_id, alg_id, 3) = eval_test.auc;
            case false
                eval_results(scale_id, alg_id, 2) = eval_test.rmse;
                eval_results(scale_id, alg_id, 3) = eval_test.mae;
                eval_results(scale_id, alg_id, 4) = eval_test.nmse;
                eval_results(scale_id, alg_id, 5) = eval_test.ev;
        end
    end
end
save(['results/B_msn_varyDataScale.mat'],'eval_results','hp');

