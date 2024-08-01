clear
close all
addpath('func','eval')
rng('default')

% setting
select_alg  = 'msn';       % msn, msn-a, msn-boost, ridge, lasso
hp.cla_flag = false;       % true: classification, false: regression
hp.num_cluster = 4;        % number of data cluster for boosting
num_Ntr_syn = 60;         % number of synthetic samples
hp.flag_graph = true;      % true: use graph info. O(dn) complexity; false: full column groups O(dn^2) complexity

% hyperparameters
hp.k       = 1;
hp.p       = 1;
hp.mu      = 0.0;
hp.lambda1 = 1e-1;
hp.lambda2 = 1e-1;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 100;
hp.inner_iter_max = 1000;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.init_flag      = true;     % true: kmeans+ridge; false: ridge only
hp.num_K          = 5;         % for knn

% load data
[data, target, model] = generate_syndata_PL(num_Ntr_syn, hp.cla_flag);
[data] = preprocess_data(data);

% main function
tic;
switch select_alg
    case 'lasso'
        [w_global, ~] = lasso(data.X_train,target.y_train,'Lambda',hp.lambda1,'MaxIter',hp.outer_iter_max);
    case 'ridge'
        w_global = base_learner(data.X_train, target.y_train, hp.lambda1, hp.cla_flag);
    case 'msn-a'
        [W_local, eval] = MSN_Lasso_fixedA(data, target, hp);
    case 'msn'
        [W_local, eval] = MSN_Lasso(data, target, hp);
    case 'msn-boost'
        [w_global, W_local] = MSN_Lasso_boost(data, target, hp);
end
toc;

% evaluation
switch select_alg
    case {'msn','msn-a'}
        eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, [], W_local, hp);
    case 'msn-boost'
        eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, w_global, W_local, hp);
    case {'lasso','ridge'}
        eval_test = evaluate_STL(data.X_test, target.y_test, w_global, hp.cla_flag);
end
if hp.cla_flag
    disp(["Testing: ACC=",num2str(eval_test.acc),", AUC=",num2str(eval_test.auc)]);
else
    disp(["Testing: RMSE=",num2str(eval_test.rmse),", MAE=",num2str(eval_test.mae)]);
end

% illustration
if strcmp(select_alg,'msn') || strcmp(select_alg,'msn-a') || strcmp(select_alg,'msn-boost')
    h1 = show_W(model.W_train, 'Ground truth W');
    h2 = show_W(W_local(1:end-1,:), 'Learned W');
end
if strcmp(select_alg,'msn') || strcmp(select_alg,'msn-a')
    figure
    plot(eval.obj)
end
