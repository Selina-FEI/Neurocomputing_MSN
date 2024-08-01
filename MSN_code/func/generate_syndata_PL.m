function [data, target, model] = generate_syndata_PL(num_N, cla_flag)
%GENERATE_SYNDATA Summary of this function goes here
%   Detailed explanation goes here

num_Cgroup = 4;    % number of data clusters
num_Rgroup = 3;    % number of feature clusters
% num_Ntr = 50;     % number of training data per cluster
num_Nva = 50;     % number of validation data per cluster
num_Nte = 50;     % number of testing data per cluster
per_fea = 0.5;    % percentage of zero rows
data_sigma = 1;   % data variance
model_sigma = 1;  % weight variance
noise_sigma = 1e-2; % noise variance
y_bias = 0;

num_Ntr = round(num_N / num_Cgroup);

% set #groups and group-specific mean
mu_max = 5;       % positions of data clusters

mu_Cgroup = linspace(-mu_max,mu_max,num_Cgroup);  % group mean array
Rgroup_list = round(linspace(3,9,num_Rgroup));
num_D = sum(Rgroup_list);    % number of dimensionality

% data sampling + group-specific model generation
X_train = cell(num_Cgroup,1);
X_val   = cell(num_Cgroup,1);
X_test  = cell(num_Cgroup,1);
W_star  = cell(num_Cgroup,1);
for k = 1 : num_Cgroup
    mean_k = mu_Cgroup(k);
    X_train{k} = mySample(num_Ntr, num_D, mean_k, data_sigma);
    X_val{k}   = mySample(num_Nva, num_D, mean_k, data_sigma);
    X_test{k}  = mySample(num_Nte, num_D, mean_k, data_sigma);
    W_star{k}  = mySample(num_D, 1, 0, model_sigma);
end
X_train = cell2mat(X_train);
X_val   = cell2mat(X_val);
X_test  = cell2mat(X_test);

% construct the ground truth weight matrix W_star
W_train = cell(1,num_Cgroup);
W_val   = cell(1,num_Cgroup);
W_test  = cell(1,num_Cgroup);
for k = 1 : num_Cgroup
    W_train{k} = W_star{k} * ones(1,num_Ntr);
    W_val{k}   = W_star{k} * ones(1,num_Nva);
    W_test{k}  = W_star{k} * ones(1,num_Nte);
end
W_train = cell2mat(W_train);
W_val   = cell2mat(W_val);
W_test  = cell2mat(W_test);

% assign row-wise sparsity
count = 1;
Rgroup_len = length(Rgroup_list);
for k = 1 : Rgroup_len
    num_zero = floor(Rgroup_list(k)*per_fea);
    W_train(count:count+num_zero, :) = 0;
    W_val(count:count+num_zero, :)   = 0;
    W_test(count:count+num_zero, :)  = 0;
    count = count + Rgroup_list(k);
end

% target generation
X_train_td = num2cell(X_train,2);
y_train    = blkdiag(X_train_td{:}) * W_train(:) + mySample(num_Ntr*num_Cgroup,1,0,noise_sigma) + y_bias;
X_val_td   = num2cell(X_val,2);
y_val      = blkdiag(X_val_td{:}) * W_val(:) + mySample(num_Nva*num_Cgroup,1,0,noise_sigma) + y_bias;
X_test_td  = num2cell(X_test,2);
y_test     = blkdiag(X_test_td{:}) * W_test(:) + mySample(num_Nte*num_Cgroup,1,0,noise_sigma) + y_bias;

if cla_flag
    y_train = sigmoid(y_train);
    y_train(y_train>0.5)  = 1;
    y_train(y_train<=0.5) = 0;
    y_val = sigmoid(y_val);
    y_val(y_val>0.5) = 1;
    y_val(y_val<=0.5) = 0;
    y_test = sigmoid(y_test);
    y_test(y_test>0.5) = 1;
    y_test(y_test<=0.5) = 0;
end

data.X_train   = X_train;
data.X_val     = X_val;
data.X_test    = X_test;
target.y_train = y_train;
target.y_val   = y_val;
target.y_test  = y_test;
model.W_train = W_train;
model.W_val   = W_val;
model.W_test  = W_test;

end

%% Define the sampling function (X \sim N(mu,sigma^2))
function X = mySample(dim1,dim2,mu,sigma)
X = sigma*randn(dim1,dim2)+mu;  
end 