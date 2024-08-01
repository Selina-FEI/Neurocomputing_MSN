%% calculate_error_reg: function description
function [mae] = calculate_MAE_reg(X_test, y_test, w)
y_pred = X_test * w;
mae = sum(abs(y_pred-y_test)) / length(y_test);
end
