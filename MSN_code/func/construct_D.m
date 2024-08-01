function [D] = construct_D(X_train, mu, num_K, flag_graph)

[num_N, num_D] = size(X_train);
if flag_graph
    num_G1 = num_N * num_K;
else
    num_G1 = num_N * (num_N-1) / 2;
end
num_G2 = num_D - 1;    % remove the bias from the penalty of row-wise sparisty 
I_d = speye(num_D);
I_n = speye(num_N);
E1 = mu .* I_n;
E2 = (1-mu) .* I_d;
D1 = cell(num_G1, 1);
D2 = cell(num_G2, 1);
index = 1;

% construct knn graph for efficient computation
if flag_graph && mu ~= 0
    [neighbors, ~] =  knnsearch(X_train, X_train, 'K', num_K+1);   % num_K+1 for including each sample in neighbors
    neighbors(:,1) = [];
end

if mu == 0
    for k = 1 : num_G2
        D2{k} = kron(I_n, E2(k,:));
    end
    D = D2;
elseif mu == 1
    for i = 1 : num_N
        if flag_graph
            for j = 1 : num_K    % exclude i from neighbors
                D1{index} = kron((E1(neighbors(i,j),:)-E1(i,:)), I_d);
                index = index + 1;
            end
        else
            for j = 1 : i
                if i ~= j
                    D1{index} = kron((E1(j,:)-E1(i,:)), I_d);
                    index = index + 1;
                else
                    continue;
                end
            end
        end
    end
    D = D1;
else
    for i = 1 : num_N
        if flag_graph
            for j = 1 : num_K
                D1{index} = kron((E1(neighbors(i,j),:)-E1(i,:)), I_d);
                index = index + 1;
            end
        else
            for j = 1 : i
                if i ~= j
                    D1{index} = kron((E1(j,:)-E1(i,:)), I_d);
                    index = index + 1;
                else
                    continue;
                end
            end
        end
    end
    for k = 1 : num_G2
        D2{k} = kron(I_n, E2(k,:));
    end
    D = [D1; D2];
end

end