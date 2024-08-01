%% calculate_error_reg: function description
function [rmse] = calculate_RMSE_reg(X_test, y_test, w)
%     whole_y = [];
%     whole_predicted = [];
%     rmse_per_task = zeros(num_task,1);

    y_predicted = X_test * w;
    rmse = sqrt(sum((y_predicted-y_test).^2)/size(y_test,1));
%     whole_y = [whole_y; y_test];
%     whole_predicted = [whole_predicted; y_predicted];
% 
%     N = length(whole_y);
%     rmse = sqrt(sum((whole_predicted-whole_y).^2)/N);
end
