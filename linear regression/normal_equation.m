% Computes the value of theta using normal equations for linear regression model 
function [theta]= normal_equation(X, y)

theta= pinv(X' * X)* X' * y;

end
