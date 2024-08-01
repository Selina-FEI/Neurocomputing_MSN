clc
clear all
addpath('results')
addpath(genpath('C:\Users\DELL\Desktop\MSN_Lasso'))
rng('default')

select_alg = 'MSN-boost';  % 'MSN' or 'MSN-boost'
data_list = {'mnist10', 'cifar', 'awa50'};

hp.k       = 2;
hp.p       = 2;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 50;
hp.inner_iter_max = 100;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.init_flag      = false;     % true: kmeans+ridge; false: ridge only
hp.num_K          = 5;         % for knn

for data_id = 1 : length(data_list)
    dataset = data_list{data_id};
    parfor fold_id = 1 : 10
        if fold_id == 10
            dataset_k = [dataset,'_',num2str(fold_id)];
        else
            dataset_k = [dataset,'_0',num2str(fold_id)];
        end
        disp(dataset_k);
        D_val_and_store_MSN_cla(dataset_k, ['D_log_',dataset_k,'_',select_alg,'.mat']);
    end
end
     