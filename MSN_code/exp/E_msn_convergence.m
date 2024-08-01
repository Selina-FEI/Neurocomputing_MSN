clear
close all
addpath('func','eval','draw','data')
addpath(genpath('data'))
rng('default')

% setting
dataset     = 'syn';       % 'syn', 'computer', 'parkinsons', 'rf1', 'sarcos', 'water'
hp.cla_flag = false;       % true: classification, false: regression
hp.num_cluster = 4;        % number of data cluster for boosting
num_Ntr_syn = 50;

% hyperparameters
hp.k       = 1;
hp.p       = 1;
hp.mu      = 0.0;
% hp.rho     = 1e-4;
hp.lambda1 = 1e-1;
hp.lambda2 = 1e-1;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 50;
hp.inner_iter_max = 100;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.init_flag      = true;     % true: kmeans+ridge; false: ridge only
hp.num_K          = 5;         % for knn

% load data
if strcmp(dataset,'syn')
    [data, target, model] = generate_syndata_PL(num_Ntr_syn, hp.cla_flag);
    [data] = preprocess_data(data);
else
    load([dataset,'_01'])
    data.X_train   = trSet.X;
    target.y_train = trSet.Y;
    data.X_val     = vaSet.X;
    target.y_val   = vaSet.Y;
    data.X_test    = teSet.X;
    target.y_test  = teSet.Y;
    clear trSet teSet vaSet
    [data] = preprocess_MTL(data);
end

% main function
tic;
[W_local, eval] = MSN_Lasso(data, target, hp);
toc;

plot(1:length(eval.obj), eval.obj, '-o','linewidth',2,'markersize',14);
set(gca,'fontsize',20);
xlabel({'Number of Iterations'},'fontsize',24);
ylabel({'Objective Value'},'fontsize',24);
grid on 

saveas(fig,['E_msn_convergence_',dataset,'.png']);
