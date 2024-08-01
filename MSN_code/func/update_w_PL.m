function [w, obj_w] = update_w_PL(w, Dt_mat, X_train, y_train, hp)  % X_train is block diagonal
%UPDATE_B_MSN Summary of this function goes here
%   Detailed explanation goes here

alpha = 0.1;   % (0,0.5]    % reduce its value if stepwise is too small
beta  = 0.1;   % (0,1)      % reduce its value if too slow
p = hp.p;
lambda2 = hp.lambda2;
eta = hp.eta;
stepsize_flag = hp.stepsize_flag;
absTol = hp.absTol;
inner_iter_max = hp.inner_iter_max;
rho = 0.01 / hp.num_G;
% rho = hp.rho;
obj_w = [];
obj_w = cat(1,obj_w,obj_value_w(X_train, y_train, w, Dt_mat, hp));
stepsize = eta;
if p == 2
    DtD = Dt_mat'*Dt_mat;
end
for iter = 1 : inner_iter_max
    % 1. calculate the gradient
    w_grad1 = gradient_for_loss(w, X_train, y_train, hp.cla_flag);
    switch p
        case 1
            c = proj_op(Dt_mat*w, lambda2, rho);
            w_grad2 = lambda2 * (Dt_mat'*c);
        case 2
            w_grad2 = (2*lambda2) * (DtD*w);    
    end
    w_grad = w_grad1 + w_grad2;

    % 2. GD via backtracking line search
    switch stepsize_flag
        case 'line'
            w_prev = w;
            while true
                w = w_prev - stepsize * w_grad;
                if obj_value_w(X_train, y_train, w, Dt_mat, hp) > ...
                        obj_value_w(X_train, y_train, w_prev, Dt_mat, hp) - alpha*stepsize*(w_grad'*w_grad)
                    stepsize = stepsize * beta;
                else
                    break;
                end
            end
        case 'fixed'
            w = w - stepsize * w_grad;
    end

    % Check the convergence condition
    obj_w = cat(1,obj_w,obj_value_w(X_train, y_train, w, Dt_mat, hp));
    if iter > 1 && abs(obj_w(iter)-obj_w(iter-1)) / obj_w(iter-1) < absTol
        break;
    end
end
end

function [obj] = obj_value_w(X_train, y_train, w, Dt_mat, hp)
obj = sum_loss_STL(X_train, y_train, w, hp.cla_flag) + hp.lambda2 * norm(Dt_mat*w, hp.p)^hp.p;
end