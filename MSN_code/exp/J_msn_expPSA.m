clear
close all
rng('default')

%% Set global parameter
dataset = 'Reuters';
method = 'MSN';     % 'agile' or 'iagile''
cla_flag = true;       % classification (true) or regression (
num_Fold = 10;

val_set = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4];

hp.k       = 2;
hp.p       = 2;
hp.mu      = 0.0;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 50;
hp.inner_iter_max = 100;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.init_flag      = true;     % true: kmeans+ridge; false: ridge only
hp.num_K          = 5;         % for knn
hp.cla_flag       = cla_flag;

if cla_flag
    num_Met = 2;
    met_set = {'ACC', 'AUC'};
else
    num_Met = 4;
    met_set = {'RMSE', 'MAE', 'NMSE', 'EV'};
end

%% Evaluation of preformance
finalMean = zeros(val_len,val_len,num_Met);
finalStd = zeros(val_len,val_len,num_Met);
for val1_id = 1 : val_len
    for val2_id = 1 : val_len
        hp.lambda1 = val_set{val1_id};
        hp.lambda2 = val_set{val2_id};
        disp([num2str(val_set{val1_id}),'-',num2str(val_set{val2_id}),...
            ' -- ',num2str(count/val_len^2)])
        tmp_results = zeros(num_Met,num_Fold);
        for fold_id = 1 : num_Fold
            if fold_id == 10
                dataset_k = [dataset,'_',num2str(fold_id)];
            else
                dataset_k = [dataset,'_0',num2str(fold_id)];
            end
            [data] = preprocess_data(data);
            [W_local, ~] = MSN_Lasso(data, target, hp);
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
        finalMean(val1_id,val2_id,:) = squeeze(mean(tmp_results,2));
        finalStd(val1_id,val2_id,:)  = squeeze(std(tmp_results,0,2) / sqrt(size(tmp_results,2)));
    end
end

%% Save the results
save(['results/J_expPSA_',dataset,'.mat'],'finalMean','finalStd','val_set','hp')
