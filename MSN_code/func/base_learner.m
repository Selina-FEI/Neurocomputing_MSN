%% learn_W_STL_cla: function description
function [w] = base_learner(X_train, y_train, lambda, cla_flag)
% base learner for classification (logistic loss) and regression (squared loss)
% min_w Loss(X,y,w) + lambda/2 * ||w||_2^2

[num_N,num_D] = size(X_train);
if cla_flag
    alpha = 1e+1; % initial step size
    EPS_G = 1e-6;
    iter_w = 0;
    iter_max = 1000; % maximal iteration numbers
    norm_g_column = Inf;
    w = zeros(num_D, 1);
    grad = zeros(num_D, 1);
    while (norm_g_column > EPS_G) && (iter_w < iter_max)
        prev_grad = grad;
        grad_loss = (X_train'*(sigmoid(X_train*w)-y_train)) / num_N;
        grad_reg = lambda * w;
        grad = grad_loss + grad_reg;
        if iter_w < 2
            stepsize = alpha;
        else
            delta_g = grad - prev_grad;
            delta_w = w - prev_w;
            stepsize = (delta_w'*delta_g)/(delta_g'*delta_g);
        end
        prev_w = w;
        w = w - stepsize*grad;
        iter_w = iter_w + 1;
        norm_g_column = sqrt(sum(grad.^2,1));
    end
else
    w = (X_train'*X_train+lambda*eye(num_D)) \ (X_train'*y_train);
end


end
