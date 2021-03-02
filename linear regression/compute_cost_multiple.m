% Compute cost for linear regression with multiple variables
function J= compute_cost_multi(X, y, theta)

m= length(y); % number of training examples

h= X*theta;
J= 1/(2*m) * (sum((h-y).^2));

end
