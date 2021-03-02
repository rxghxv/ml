% Compute cost function and gradient for logistic regression with regularization
function [J, grad]= cost_function_reg(theta, X, y, lambda) 

m= length(y); % number of training examples

h= sigmoid(X*theta);
cost= y'*log(h)+(1-y)'*log(1-h);
J= (-1/m)*cost + (lambda/(2*m))*(sum(theta(2:end,1).^2));

grad= ((1/m)*(X'*(h-y))) + ((lambda/m)*theta);
a= (1/m)*(X'*(h-y));
grad(1,1)= a(1,1); % because theta_0 should not be paramterised

end
