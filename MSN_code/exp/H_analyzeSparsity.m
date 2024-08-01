function [sparsity] = H_analyzeSparsity(W_local)
%H_ANALYZESPARSITY Summary of this function goes here
%   Detailed explanation goes here

th = 0.1;
sparsity = zeros(3,1);
[num_D, num_N] = size(W_local);

% row sparsity
row_values = zeros(num_D,1);
for j = 1 : num_D
    row_values(j) = norm(W_local(j,:),2);
end
row_mean = mean(row_values);
th_row = th * row_mean;
row_values(row_values <= th_row) = 0;
row_sparsity = 1 - nnz(row_values) / num_D;

% column-pairwise sparsity
col_values = zeros(num_N,num_N);
for i = 1 : num_N
    for j = 1 : num_N
        col_values(i,j) = norm(W_local(:,i)-W_local(:,j),2);
    end
end
col_mean = mean(col_values);
th_col = th * col_mean;
col_values(col_values <= th_col) = 0;
col_sparsity = 1 - nnz(col_values) / (num_N*num_N);

% element-wise sparsity
ele_values = abs(W_local);
ele_mean = mean(ele_values(:));
th_ele = th * ele_mean;
ele_values(ele_values <= th_ele) = 0;
ele_sparsity = 1 - nnz(ele_values) / (num_D*num_N);

sparsity(1) = row_sparsity;
sparsity(2) = col_sparsity;
sparsity(3) = ele_sparsity;

end

