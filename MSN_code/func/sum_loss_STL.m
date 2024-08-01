%% sum_loss for both STL and MLC
function [outputs] = sum_loss_STL(X_train, y_train, w, cla_flag)
outputs = 0;
if cla_flag
    p_t = sigmoid(X_train * w);
    outputs = outputs - 1 / size(y_train, 1) * (y_train' * log(p_t) + (1 - y_train)' * log(1 - p_t));
    outputs = sum(outputs);
else
    outputs = outputs + 0.5 / size(y_train, 1) * norm(X_train * w - y_train) ^ 2;
end
end