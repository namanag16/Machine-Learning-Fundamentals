function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

Xnew = X';
Ynew = y';  % size(Ynew) = 1 x m | m col's for m training egs 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

## for linear regression grad was => grad = sum(repmat(theta'*Xnew-Ynew,size(Xnew,1),1).*Xnew,2)/m;   


htheta = sigmoid(X*theta); ## X is m x (n+1) and theta n+1 x 1 | n = #features = 28 | htheta is m x 1

J = (sum((-y.*log(htheta)) - ((1-y).*log(1-htheta)))/m) + (theta(2:end,:)' * theta(2:end,:))*(lambda/(2*m));  # J is a scaler 

grad = sum(repmat((htheta - y)',size(Xnew,1),1).*Xnew,2)/m + [0;theta(2:end,:)]*(lambda/m);

## grad = grad'; ## convert to column vector | grad is now 28 x 1 (n+1 x 1)
## size(grad)



% =============================================================

end
