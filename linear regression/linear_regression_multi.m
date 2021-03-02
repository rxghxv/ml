%% Linear Regression Model with Multiple Variables
%
% files used
%     plot_data.m
%     gradient_descent.m
%     compute_cost.m
%     gradient_descent_multi.m
%     compute_cost_multi.m
%     feature_normalization.m
%     normal_equation.m
%

%% ================ Part 1: Feature Normalization ================

%% Clear and Close Figures
clear; close all; clc
fprintf('Loading data ...\n');

%% Load Data
data= load('data2.txt');
X= data(:, 1:2);
y= data(:, 3);
m= length(y);

fprintf('Program paused. Press any key to continue.\n');
pause;

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

[X mu sigma]= feature_normalization(X);

% Add intercept term to X
X= [ones(m, 1) X];


%% ================ Part 2: Gradient Descent ================
fprintf('Running gradient descent ...\n');

% Choose some alpha value and number of gradient steps
alpha= 0.01;
num_iters= 400;

% Init Theta and Run Gradient Descent 
theta= zeros(3, 1);
[theta, J_history] = gradient_descent_multiple(X, y, theta, alpha, num_iters);

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('Theta computed from gradient descent: \n');
fprintf(' %f \n', theta);
fprintf('\n');

% Estimate the price of a 1650 sq-ft, 3 br house
% NOTE: The first column of X is all-ones. Thus, it does not need to be normalized.
price= [1, 1650, 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using gradient descent):\n $%f\n'], price);

fprintf('Program paused. Press any key to continue.\n');
pause;

%% ================ Part 3: Normal Equations ================
fprintf('Solving with normal equations...\n');

%% Load Data
data= csvread('data2.txt');
X= data(:, 1:2);
y= data(:, 3);
m= length(y);

% Add intercept term to X
X= [ones(m, 1) X];

% Calculate the parameters from the normal equation
theta= normal_equation(X, y);

% Display normal equation's result
fprintf('Theta computed from the normal equations: \n');
fprintf(' %f \n', theta);
fprintf('\n');


% Estimate the price of a 1650 sq-ft, 3 br house
price= [1, 1650, 3] * theta;

fprintf(['Predicted price of a 1650 sq-ft, 3 br house ' ...
         '(using normal equations):\n $%f\n'], price);

