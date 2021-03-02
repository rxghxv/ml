%% Logistic Regression Using Regularization
%
% files used
%     sigmoid.m
%     cost_function.m
%     predict.m
%     cost_function_reg.m
%     map_feature.m
%

%% Initialization
clear; close all; clc

%% Load Data
%  The first two columns contains the X values and the third column contains the label (y)
data= load('data2.txt');
X= data(:, [1, 2]); y = data(:, 3);

plot_data(X, y);

hold on;
xlabel('Microchip Test 1')
ylabel('Microchip Test 2')
legend('y = 1', 'y = 0')
hold off;

%% =========== Part 1: Regularized Logistic Regression ============
% Add Polynomial Features

% NOTE: map_feature also adds a column of ones for us, so the intercept term is handled
X= map_feature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta= zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda= 1;

% Compute and display initial cost and gradient for regularized logistic regression
[cost, grad]= cost_function_reg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));

fprintf('\nProgram paused. Press any key to continue.\n');
pause;

% Compute and display cost and gradient with all-ones theta and lambda= 10
test_theta= ones(size(X,2),1);
[cost, grad]= cost_function_reg(test_theta, X, y, 10);

fprintf('\nCost at test theta (with lambda = 10): %f\n', cost);
fprintf('Gradient at test theta - first five values only:\n');
fprintf(' %f \n', grad(1:5));

fprintf('\nProgram paused. Press any key to continue.\n');
pause;

%% ============= Part 2: Regularization and Accuracies =============
% Trying different values of lambda and seeing how regularization happens
% Trying values of lambda (0, 1, 10, 100).
%
% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (can be varied)
lambda= 1;

% Set Options
options= optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag]= ...
	fminunc(@(t)(cost_function_reg(t, X, y, lambda)), initial_theta, options);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
