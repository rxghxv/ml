%Compute our cost function for linear regression
function J= compute_cost(X, y, theta) 

m= length(y); % number of training examples

h= X*theta; %hypothesis
J= 1/(2*m) * (sum((h-y).^2)); % compute the cost function J at some particular value of theta

end