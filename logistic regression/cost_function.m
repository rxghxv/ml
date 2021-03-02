% Compute cost function and gradient for logistic regression
function [J, grad]= cost_function(theta, X, y)

m= length(y); % number of training examples

h= sigmoid(X*theta);
cost= y'*log(h)+(1-y)'*log(1-h);
J= (-1/m)*cost;

grad= (1/m)*(X'*(h-y));

end
