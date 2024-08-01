function g = sigmoid(z)

%sigmoid function
%z = w'*x

g = 1./(1+exp(-z));

g(find(g > 0.999)) = 0.999;
g(find(g < 0.001)) = 0.001;

end
