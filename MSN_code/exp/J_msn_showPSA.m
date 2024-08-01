clc
clear
close all
addpath('results');

method  = 'MSN';
dataset = 'NUS-Object';

if cla_flag
    num_Met = 2;
    met_set = {'ACC', 'AUC'};
else
    num_Met = 4;
    met_set = {'RMSE', 'MAE', 'NMSE', 'EV'};
end
val_set  = num2cell(-4:4);

val_len  = length(val_set);
met_len  = length(met_set);

file_info = ['results/J_expPSA_',dataset,'.mat'];
load(['results/',file_info])
y_label = '$\lambda_2$';
x_label = '$\labmda_1$';

% Visualize the results -- color map
for met_id = 1:met_len
    this_met = met_set{met_id};
    h = figure('pos',[100 100 600 520]);
    tmpRes = squeeze(finalMean(:,:,met_id));
    imagesc(tmpRes);
    colorbar;
    set(gca,'XTickLabel',val_set, 'XTick',1:val_len,'Fontsize',24);
    set(gca,'YTickLabel',val_set, 'YTick',1:val_len,'Fontsize',24);
    xlabel(x_label,'Interpreter','LaTex','Fontsize',36);
    ylabel(y_label,'Interpreter','LaTex','Fontsize',36);
    title(met_set{met_id},'Fontsize',18);
    colormap(jet);
    saveas(h,[file_info,'_',this_met,'.png']);
end