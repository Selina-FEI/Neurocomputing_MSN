% ER
clc
clear all

cla_flag = true;
data_set = {'mnist10', 'cifar', 'awa50'};
mu_set = 0 : 0.1: 1;
mu_len = length(mu_set);
data_len = length(data_set);

if cla_flag
    num_Met = 2;
    met_set = {'ACC', 'AUC'};
else
    num_Met = 4;
    met_set = {'RMSE', 'MAE', 'NMSE', 'EV'};
end

hp.k       = 2;
hp.p       = 2;
hp.eta     = 1e-1;
hp.absTol  = 1e-4;
hp.outer_iter_max = 50;
hp.inner_iter_max = 100;
hp.stepsize_flag  = 'line';    % line (line search) or fixed (stepsize = hp.eta)
hp.warmup_flag    = true;      % true: initialization; false: random initialization
hp.init_flag      = true;      % true: kmeans+ridge; false: ridge only
hp.num_K          = 5;         % for knn
hp.cla_flag = cla_flag;       % true: classification, false: regression

comp_results = zeros(data_len, mu_len, num_Met);
comp_sparsity = zeros(data_len, mu_len, 2);
for data_id = 1 : length(data_set)
    dataset = data_set{data_id};
    for mu_id = 1 : mu_len
        hp.mu = mu_set(mu_id);
        tmp_results = zeros(num_Met, num_Folds);
        tmp_sparsity = zeros(3, num_Folds);    % row-sparsity and column-pairwise sparsity
        for fold_id = 1 : 10
            if fold_id == 10
                dataset_k = [dataset,'_',num2str(fold_id)];
            else
                dataset_k = [dataset,'_0',num2str(fold_id)];
            end
            [data] = preprocess_data(data);
            [W_local, ~] = MSN_Lasso(data, target, hp);
            eval_test = evaluate_PL(data.X_test, target.y_test, data.X_train, [], W_local, hp);
            if cla_flag
                tmp_results(1, fold_id) = eval_test.acc;
                tmp_results(2, fold_id) = eval_test.auc;
            else
                tmp_results(1, fold_id) = eval_test.rmse;
                tmp_results(2, fold_id) = eval_test.mae;
                tmp_results(3, fold_id) = eval_test.nmse;
                tmp_results(4, fold_id) = eval_test.ev;
            end
            tmp_sparsity(:, fold_id) = H_analyzeSparsity(W_local);
        end
        comp_results(data_id, mu_id, :) = mean(tmp_results, 2);
        comp_sparsity(data_id, mu_id, :) = mean(tmp_sparsity, 2);
    end
end
save(['results/G_msn_multiLevel.mat'], 'hp', 'comp_results');


data_id = 1;
this_results = comp_results(data_id,:,:);
for met_id = 1 : num_Met
    plot(1:mu_len, this_results(:,met_id), '-^','linewidth',2,'markersize',14);
    set(gca,'fontsize',20);
    xlabel({'$\mu$'},'fontsize',24, 'interpreter', 'latex');
    ylabel(met_set{met_id},'fontsize',24);
    grid on
    saveas(fig,['H_msn_evalMu_',met_set{met_id},'.png']);
end
figure
this_sparsity = comp_sparsity(data_id,:,:);
plot(this_sparsity(:,1), '-^', 'LineWidth', 2, 'Markersize', 14); 
hold on
plot(this_sparsity(:,2), '-v', 'LineWidth', 2, 'Markersize', 14);
hold on
plot(this_sparsity(:,3), '-d', 'LineWidth', 2, 'Markersize', 14);
grid on
set(gca,'fontsize',20);
xlabel({'$\mu$'},'fontsize',24, 'interpreter', 'latex');
ylabel('Sparsity','fontsize',26);
[~,objh] = legend(['Row wise','Column pairwise', 'Element wise'],'Location','northwest','fontsize',18);
objhl = findobj(objh, 'type', 'line'); %// objects of legend of type line
set(objhl, 'Markersize', 14); %// set marker size as desired

% for met_id = 1 : num_Met
%     b = bar(comp_resutls(:,:,met_id));
%     set(gca,'xticklabel', data_set, 'FontSize', 22,'fontweight','bold' )
%     ylabel(met_set{met_id}, 'FontSize', 22, 'fontweight','bold')
%     legend('Fixed \alpha','Update \alpha', 'FontSize', 22);
%     % legend('Fixed \alpha','Update \alpha', 'FontSize', 22, 'fontweight','bold', 'interpreter', 'latex');
%     saveas(fig,['H_msn_evalMu_',met_set{met_id},'.png']);
% end
