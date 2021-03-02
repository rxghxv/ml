% Performs gradient descent method for multiple variables
function [theta, J_history]= gradient_descent_multiple(X, y, theta, alpha, num_iters)

m= length(y); % number of training examples

for iter= 1:num_iters
    h= X*theta;
    theta= theta- (alpha/m) * (X' * (h-y)); 
    J_history(iter)= compute_cost_multiple(X, y, theta);
end

end
