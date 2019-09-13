function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1); ## = number of training/testing ex
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

## since we don't know number of hidden layers, we ll write non scalabale code as below 

X = [ones(size(X,1),1) X];
Xnew = X';
z1 = Theta1 * Xnew ;  ## 25*401 x 401*m = 25 * m;
a2 = sigmoid(z1);
z2 = Theta2 * [ones(1, size(a2,2));a2] ; ## 10*26 x 26*m = 10*m;
a3 = sigmoid(z2);

[m,p] = max(a3);
p=p';




% =========================================================================


end
