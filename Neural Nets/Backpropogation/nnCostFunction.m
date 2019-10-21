function [J grad] = nnCostFunction(nn_params, ...             	## unrolled params containing all theta as column vector
                                   input_layer_size, ...		## number of inputs
                                   hidden_layer_size, ...		## we consider only one hidden layer | size = #units in it
                                   num_labels, ...				## = k, number of output labels | 
                                   X, ...						## m*n matrix| n= #features= input_layer_size -1| m=# training examples
								   y, ...						## output values m*1| need to change each to m*k 
								   lambda)    					## regularization rate
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
## however #hidden layers are fixed = 1 ( we will assume it as a variable)
hlcount = 1;
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% works for dataset of any size and any #labels k (k>=3)

ynew = zeros(m, num_labels);  ## m*k
for i =1:m
	ynew(i,y(i)) = 1;
end



a = X';   					% a = n * m ("a" considered 3D matrix | each third dimension = one hidden layer)
mul = {a}; 					% cell used to store a's of diff sizes and still get 3D structure
theta = {Theta1,Theta2};	% theta's here are not 3D | w/o that code can't be modular for #hidden layers 
regterm = 0;

for i = 1:hlcount + 1 
	tz = theta{i} * [ones(1,m); mul{i}]; ## for i= 1 => 25*401 x 401*m = 25 * m | and so on 
	ta = sigmoid(tz);
	mul{i+1} = ta;
	regterm = regterm + sum(sum(theta{i}(:,2:end).^2));
end


yy = ynew';
hth = mul{hlcount+2};										% size(mul{hlcount+2}) ##  = k*m | ynew' = k*m
regterm = (lambda/(2*m))*regterm;
J = sum(sum(yy.*log(hth) + (1-yy).*log(1-hth)))/(-m) + regterm;



%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

% del3 = hth-yy																			% 10*m
% del2 = (Theta2'*del3)(2:end,:) .* sigmoidGradient(Theta1 * [ones(1,m); mul{1}])		% 25*m

Del = {zeros(hidden_layer_size, input_layer_size+1), zeros(num_labels, hidden_layer_size+1)};	% Del dimensions include bias term
X = [ones(size(X,1),1) X]; 			% m * n+1

for i=1:m
	a1 = X(i,:);			% 1 * n+1 | already has bias unit
	% Xnew = X';
	z2 = Theta1 * a1' ;  ## 25*401 x 401*m = 25 * m;
	a2 = sigmoid(z2);
	z3 = Theta2 * [ones(1, size(a2,2));a2] ; ## 10*26 x 26*m = 10*m;
	a3 = sigmoid(z3);
	
	% del3==dz2 in DL C1W3 (Planar_data_classification_with_onehidden_layer_v6c ) = def backward_propagation(parameters, cache, X, Y)
	% del2==dZ1
	% batch processing done instead of individual egs in DL to save time
	del3 = a3 - ynew(i,:)';     								% 10*1							
	del2 = (Theta2'*del3)(2:end,:) .* sigmoidGradient(z2);		% 25*1
	
	% Del1=dW1 in DL C1W3 (Planar_data_classification_with_onehidden_layer_v6c ) = def backward_propagation(parameters, cache, X, Y)
	% Del2=dW2
	Del{1} = Del{1} + del2 * a1;   								% 25*401 | included bias term when doing del
	Del{2} = Del{2} + del3 * [ones(1, size(a2,2));a2]';			% 10*26
	
	% Theta1 = Theta1 + del2 * a1;
	% Theta2 = Theta2 + del3 * [ones(1, size(a2,2));a2]';
	
end

Theta1_grad = Del{1}/m;
Theta2_grad = Del{2}/m;


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% Theta1_grad = Theta1_grad + [zeros(size(Theta1_grad,1),1) Del{1}(:,2:end) * (lambda)];
% Theta2_grad = Theta2_grad + [zeros(size(Theta2_grad,1),1) Del{2}(:,2:end) * (lambda)];

Theta1_grad = Theta1_grad + [zeros(size(Theta1_grad,1),1) Theta1(:,2:end)*(lambda/m)];
Theta2_grad = Theta2_grad + [zeros(size(Theta2_grad,1),1) Theta2(:,2:end)*(lambda/m)];











% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
