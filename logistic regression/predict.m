% Predict whether the label is 0 or 1 using learned logistic regression parameters theta
function p= predict(theta, X)
%   Computes the predictions for X 
%   i.e., if sigmoid(theta'*x) >= 0.5, predict 1

m= size(X, 1); % Number of training examples

h= sigmoid(X*theta);
p= h>=0.5;

end
