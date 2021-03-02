% Performs gradient descent method
function [theta, J_history]= gradient_descent(X, y, theta, alpha, num_iters)
% num_iters is no. gradient steps which can be varied too

m= length(y); % number of training examples

for iter= 1:num_iters
    h= X*theta;
    theta= theta- (alpha/m) * (X' * (h-y)); 
    J_history(iter)= compute_cost(X, y, theta); % saving the value of cost function J in every iteration
end

end