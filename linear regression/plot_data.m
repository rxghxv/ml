% Plots the data points x and y
function plot_data(x, y)

figure; % opens a new figure window

plot(x, y, 'rx', 'MarkerSize', 10);
ylabel('Profit');
xlabel('Population');

end

