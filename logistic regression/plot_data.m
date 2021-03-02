% Plots the data points X and y 
function plot_data(X, y)
%   Plots the data points with + for the positive examples and
%   o for the negative examples

% Create New Figure
figure; hold on;

ones= find(y==1); zeroes= find(y==0);
plot(X(ones,1),X(ones,2),'k+');
plot(X(zeroes,1),X(zeroes,2),'ko');

hold off;

end
