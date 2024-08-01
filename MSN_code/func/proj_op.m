function [u] = proj_op(u, lambda2, rho)
%PROJ_OP Summary of this function goes here
%   Detailed explanation goes here

u = (lambda2/rho) * u;
u(u>=1)  = 1;
u(u<=-1) = -1;

end

