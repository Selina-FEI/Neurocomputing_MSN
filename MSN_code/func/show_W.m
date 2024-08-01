function [h] = show_W(W, W_disp)
%SHOW_Z Summary of this function goes here
%   Detailed explanation goes here

figure
% Ta = 0.5 *(abs(Ta) + abs(Ta'));
% Z = Z / 10;
% Ta = Ta / (max(max(Ta)) - min(min(Ta)));
W = abs(W);
W = W ./ max(W(:));
% Z(Z < 0.15) = 0;
colormap(flipud(gray));
h = imagesc(W);
colorbar;
title(W_disp)
set(gca,'fontsize',24,'fontweight','bold');


end

