function [eval_result] = evaluate_booster(eval_cell, idx_va_cell, num_Nva, cla_flag)
%EVALUATE_BOOSTER Summary of this function goes here
%   Detailed explanation goes here
num_cluster = length(eval_cell);
if cla_flag
    acc = 0;
    auc = 0;
    for k = 1 : num_cluster
        num_Nk = nnz(idx_va_cell{k});
        acc = acc + eval_cell{k}.acc * num_Nk;
        auc = auc + eval_cell{k}.auc * num_Nk;
    end
    eval_result.acc = acc / num_Nva;
    eval_result.auc = auc / num_Nva;
else
    rmse = 0;
    mae  = 0;
    nmse = 0;
    for k = 1 : num_cluster
        num_Nk = nnz(idx_va_cell{k});
        rmse = rmse + eval_cell{k}.rmse * num_Nk;
        mae  = mae + eval_cell{k}.mae * num_Nk;
        nmse = nmse + eval_cell{k}.nmse * num_Nk;
    end
    eval_result.rmse = rmse / num_Nva;
    eval_result.mae  = mae / num_Nva;
    eval_result.nmse = nmse / num_Nva;
end
end

