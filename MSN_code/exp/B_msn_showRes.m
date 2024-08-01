clear
close all

load('results/B_msn_varyDataScale.mat');

if hp.cla_flag
    met_set = {'Times (s)', 'ACC', 'AUC'};
else
    met_set = {'Times (s)', 'RMSE', 'MAE', 'NMSE', 'EV'};
end

alg_set  = {'Basic','Divide-and-Conquer'};
% met_set = {'Time (s)','AUC','Accuracy','MacroF1','MicroF1'};
% scale_set = 200:100:1000;
scale_set = 20:10:50;

alg_len = length(alg_set);
met_len = length(met_set);
scale_len = length(scale_set);
% finalMean = zeros(met_len,scale_len,alg_len);
% finalStd = zeros(met_len,scale_len,alg_len);


% for alg_id = 1 : 5
%     this_alg = alg_set{alg_id};
%     for scale_id = 1 : scale_len
%         this_scale = scale_set{scale_id};
%         load(['results\E_agile_varyDN_',flag,'_',num2str(this_scale),'_',this_alg,'_tmp.mat']);
%         finalMean(:,scale_id,alg_id) = mean_res;
%         finalStd(:,scale_id,alg_id) = std_res;
%     end
% end

fct = 0.3;
plot_mark = {'-^','-v','-s','-o','-d'};
for met_id = 1 : met_len
    fig = figure;
    hold on
%     tmp_mean = squeeze(finalMean(met_id,:,:));
%     tmp_std  = squeeze(finalStd(met_id,:,:));

    tmpRes = eval_results(2:end,:,met_id);
    zMax = max(tmpRes(:));
    zMin = min(tmpRes(:));
    zMin = zMin - fct * zMin;
    zMax = zMax + fct * zMax;
    zRange = [zMin zMax];
    
    %     plot(1:scale_len,tmpRes,'-^','linewidth',3,'markersize',12);
    for alg_id = 1 : alg_len
%         h = errorbar(1:scale_len,tmpRes(:,alg_id),tmp_std(:,alg_id),plot_mark{alg_id},'linewidth',2,'markersize',14,'capsize',14);     
%         h.CapSize = 8;        
        plot(tmpRes(1:end,alg_id),plot_mark{alg_id},'linewidth',2,'markersize',14);
%         h = errorbar(1:scale_len,tmpRes(:,alg_id),tmp_std(:,alg_id),plot_mark{alg_id},'linewidth',2,'markersize',14,'capsize',14);     
    end   
    ylim(zRange);
    grid on
    
    
    %     bar(tmp_mean);
%     for i = 1:scale_len
%         j = 1:alg_len;
%         x = -0.45 + i + 1/5.5 * j;
%         h = errorbar(x, tmp_mean(i,j), tmp_std(i,j), '.');
%         h.LineWidth = 1;
%     end
% %     str1 = {;'';'';'100';'';'';'';'';'200';'';''};
%     str1 = {'100';'200';'300';'400';'500';'600';'700';'800';'900';'1000';};    
%     str1 = {'0';'200';'400';'600';'800';'1000';};    
%     str1 = {'0';'500';'1000';};        

%     str1 = {'200','400','600','800','1000';};    
    str1 = {'100','600','1100';};
    
%     str1 = {'1';'2';'3';'4';'5';'6';'7';'8';'9';'10';};        
%     str1 = {'10%';'20%';'30%';'40%';'50%';'60%';'70%';'80%';'90%'};    
    set(gca, 'XTickLabel',str1,'fontsize',20);
%     set(gca, 'XTickLabel',scale_set);
%     xlabel(xlabel_str,'fontsize',24);
    xlabel('Number of Samples','fontsize',24);
    
%     xlabel('Number of training samples');
    ylabel(met_set{met_id},'fontsize',26);
    if met_id == 1
        [~,objh] = legend(alg_set,'Location','northwest','fontsize',18);
    else
        [~,objh] = legend(alg_set,'Location','northeast','fontsize',18);
    end
    objhl = findobj(objh, 'type', 'line'); %// objects of legend of type line
    set(objhl, 'Markersize', 14); %// set marker size as desired

%         legend(alg_set,'Location','Northeast','fontsize',14);

    
%     if met_id ==1
%         legend(alg_set,'Location','Northeast');
%     else
%         legend(alg_set,'Location','Southeast');
%     end
    grid on
%     title(['Experiments on the ',dataset,' dataset.'])
if met_id == 1
    met_set{met_id} = 'Time';
end
    saveas(fig,['B_msn_varyDataScale_',met_set{met_id},'.png']);
end