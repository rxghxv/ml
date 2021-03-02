%% Logistic Regression
%
% files used
%     sigmoid.m
%     cost_function.m
%     plot_data.m
%     predict.m
%

%% Initialization
clear; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column contains the label
data= load('data1.txt');
X= data(:, [1, 2]); y = data(:, 3);

%% ==================== Part 1: Plotting ====================
fprintf(['Plotting data with + indicating (y = 1) examples and o ' ...
         'indicating (y = 0) examples.\n']);

plot_data(X, y);

hold on;
xlabel('Exam 1 score')
ylabel('Exam 2 score')
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press any key to continue.\n');
pause;


%% ============ Part 2: Compute Cost and Gradient ============
[m, n]= size(X);

% Add intercept term to x and X_test
X= [ones(m, 1) X];

% Initialize fitting parameters
initial_theta= zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad]= cost_function(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

% Compute and display cost and gradient with non-zero theta
test_theta= [-24; 0.2; 0.2];
[cost, grad]= cost_function(test_theta, X, y);

fprintf('\nCost at test theta: %f\n', cost);
fprintf('Gradient at test theta: \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press any key to continue.\n');
pause;


%% ============= Part 3: Optimizing algorithm using fminunc  =============
%  Set options for fminunc
options= optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost]= ...
	fminunc(@(t)(cost_function(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

fprintf('\nProgram paused. Press any key to continue.\n');
pause;

%% ============== Part 4: Predict and Accuracies ==============
prob= sigmoid([1 45 85] * theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n'], prob);
     
% Compute accuracy on our training set
p= predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);
fprintf('\n');
