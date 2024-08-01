%% cal_AUC: function description
function [result] = calculate_AUC_cla(X_test, y_test, w)
% 	num_task = size(X_test, 2);
%     Yout = cell(num_task,1);
% 	for t = 1 : num_task
		y_pred = sigmoid(X_test * w);
% 	end
	result = evalAUC(y_pred, y_test);
end