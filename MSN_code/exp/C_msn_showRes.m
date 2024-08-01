clear
close all

load('results/B_msn_varyDataScale.mat');

if hp.cla_flag
    met_set = {'Times (s)', 'ACC', 'AUC'};
else
    met_set = {'Times (s)', 'RMSE', 'MAE', 'NMSE', 'EV'};
end

% scale_set = 200:100:1000;
% scale_set = 2:2:10;

% alg_len = length(alg_set);
met_len = length(met_set);
% scale_len = length(scale_set);

fct = 0.3;
plot_mark = {'-^','-v','-s','-o','-d'};
for met_id = 1 : met_len
    fig = figure;
    hold on
    tmpRes = eval_results(:,met_id);
    zMax = max(tmpRes(:));
    zMin = min(tmpRes(:));
    zMin = zMin - fct * zMin;
    zMax = zMax + fct * zMax;
    zRange = [zMin zMax];
    plot(tmpRes,plot_mark{1},'linewidth',2,'markersize',14);
    ylim(zRange);
    grid on   
    str1 = {'200','400','600','800','1000';};       
    set(gca, 'XTickLabel',str1,'fontsize',20);
    xlabel('Number of Data Clusters','fontsize',24);
    ylabel(met_set{met_id},'fontsize',26);
    [~,objh] = legend(['Divide-and-Conquer'],'Location','northwest','fontsize',18);
    objhl = findobj(objh, 'type', 'line'); %// objects of legend of type line
    set(objhl, 'Markersize', 14); %// set marker size as desired
%         legend(alg_set,'Location','Northeast','fontsize',14);
%     if met_id ==1
%         legend(alg_set,'Location','Northeast');
%     else
%         legend(alg_set,'Location','Southeast');
%     end
    grid on
    if met_id == 1
        met_set{met_id} = 'Time';
    end
    saveas(fig,['C_msn_varyNumCluster_',met_set{met_id},'.png']);
end