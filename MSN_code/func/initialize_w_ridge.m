%% single_task_learn_reg: function description
function [w_init] = initialize_w_ridge(X_train, y_train, X_val, y_val, cla_flag)
err = Inf;
acc_prev = 0;
num_Nva = length(y_val);
lambda_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1];
list_len = length(lambda_list);
for i = 1 : list_len
    w = base_learner(X_train, y_train, lambda_list(i), cla_flag);
    if cla_flag
        y_pred = sign(X_val * w);
        y_pred(y_pred == -1) = 0;
        acc = 1- sum(~(y_pred == y_val)) / num_Nva;
        if acc >= acc_prev
            acc_prev = acc;
            w_init = w;
        end
    else
        rmse = sqrt(sum((X_val*w-y_val).^2)/size(y_val,1));
        if rmse <= err
            err = rmse;
            w_init = w;
        end
    end
end
end

