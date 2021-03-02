% Normalizes the features in X
function [X_norm, mu, sigma]= feature_normalization(X)

X_norm= X;

mu= mean(X);
sigma= std(X);
X_norm= (X-mu)./sigma;

end
