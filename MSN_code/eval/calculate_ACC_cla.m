%% calculate_error_cla: function description
function [accuracy] = calculate_ACC_cla(X_test, y_test, w)

% num_task = size(X_test, 2);
% error_rate = zeros(num_task, 1);
% error_num = zeros(num_task, 1);
% total_sample = 0;
% num_N = length(y_test);
% total_sample = total_sample + N_t;
prediction = sign(X_test * w);
prediction(prediction == -1) = 0;
accuracy = 1- sum(~(prediction == y_test)) / length(y_test);
% error_num(t) = sum(~(prediction == y_test{t}));

end