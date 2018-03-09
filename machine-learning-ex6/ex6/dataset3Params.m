function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
lst = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
min_err = 1;
best_C = lst(1);
best_sigma = lst(1);


for i = 1:numel(lst)
    for j = 1:numel(lst)
        model= svmTrain(X, y, lst(i), @(x1, x2) gaussianKernel(x1, x2, lst(j)));
        predictions = svmPredict(model, Xval);
        err_val = mean(double(predictions ~= yval));
        if err_val < min_err
            min_err = err_val;
            best_C = lst(i);
            best_sigma = lst(j);
        end
    end
end

C = best_C;
sigma = best_sigma;

% =========================================================================

end
