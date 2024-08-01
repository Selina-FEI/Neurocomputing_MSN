function [eval_test] = evaluate_STL(X_test, y_test, w, cla_flag)
if cla_flag
    eval_test.acc = calculate_ACC_cla(X_test, y_test, w);
    eval_test.auc = calculate_AUC_cla(X_test, y_test, w);  % AUC
    eval_test.metric_list = {'ACC', 'AUC'};
else
    eval_test.rmse = calculate_RMSE_reg(X_test, y_test, w); %RMSE
    eval_test.mae  = calculate_MAE_reg(X_test, y_test, w); %MAE
    eval_test.metric_list = {'RMSE', 'MAE'};
end
end