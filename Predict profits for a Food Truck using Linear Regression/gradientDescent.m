function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
##Xnew = [X(:,1)';X(:,2)'];  % format X as 2 x m matrix | each col is (1,x) | m col's for m training eg
Xnew = X';
Ynew = y';  % size(Ynew) = 1 x m | m col's for m training egs 

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %







    % ============================================================
    
	% grad is 2 x 1 for univariate regression
    % rows in grad = number of parameters 
    
	grad = sum(repmat(theta'*Xnew-Ynew,size(Xnew,1),1).*Xnew,2)/m;   
    theta = theta - alpha*grad;
    ## err=sum((Ynew-theta'*Xnew).^2);
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
