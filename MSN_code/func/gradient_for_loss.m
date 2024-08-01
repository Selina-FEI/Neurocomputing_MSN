% gradient_for_loss: for both STL and MLC
function [grad_w] = gradient_for_loss(w, X_train, y_train, cla_flag)
% [num_dim, num_task] = size(w);
% grad_w = zeros(length(w), 1);
% for t = 1:num_task
    if cla_flag
        grad_w = (1 / size(y_train, 1)) * (X_train' * (sigmoid(X_train * w) - y_train));
    else
        grad_w = (1 / size(y_train, 1)) * (X_train' * (X_train * w - y_train));
    end
% end
end