% function [data,data_raw,para] = standardizeData(data,para)
% function [data,para] = standardize(data,para)

% data standization
% num_T = size(X_val,1);
num_Ntr = size(X_train,1);
num_Nva = size(X_val,1);
num_Nte = size(X_test,1);
id_tr = 1 : num_Ntr;
id_va = (1 : num_Nva) + num_Ntr;
id_te = (1 : num_Nte) + num_Ntr + num_Nva;

% for t = 1 : num_T
data_t = [X_train; X_val; X_test];
data_t = standardize(data_t);
X_train = data_t(id_tr, :);
X_val   = data_t(id_va, :);
X_test  = data_t(id_te, :);
% end

if length(unique(y_val)) <= 2
    hp.cla_flag = true;   % classification 
else
    hp.cla_flag = false;  % regression
end


function [data] = standardize(data)
% Standardize the data and add bias dimensionality to each view
    data = zscore(data);
    data = cat(2,data,ones(size(data,1),1));
end