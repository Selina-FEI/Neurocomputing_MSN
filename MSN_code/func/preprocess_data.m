function [data] = preprocess_data(data)
%PRECALRES standardize the input data and add the bias dimension

X_train = data.X_train;
X_val   = data.X_val;
X_test  = data.X_test;
num_Ntr = size(X_train,1);
num_Nva = size(X_val,1);
num_Nte = size(X_test,1);
id_tr = 1 : num_Ntr;
id_va = (1 : num_Nva) + num_Ntr;
id_te = (1 : num_Nte) + num_Ntr + num_Nva;

% standardization
X = [X_train; X_val; X_test];
X = zscore(X);          
X = cat(2,X,ones(size(X,1),1));
X_train = X(id_tr, :);
X_val   = X(id_va, :);
X_test  = X(id_te, :);

data.X_train = X_train;
data.X_val   = X_val;
data.X_test  = X_test;

end