function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
htheta = sigmoid(X*theta); ## X is m x 3 and theta 3 x 1 | 3 = n + 1 | n = #features | htheta is m x 1

J = sum((-y.*log(htheta)) - ((1-y).*log(1-htheta)))/m ;  # J is a scaler 

grad = sum(((repmat((htheta - y),1,size(X,2))).*X))/m;
grad = grad'; ## convert to column vector | grad is 3 x 1 
size(grad);






% =============================================================

end
