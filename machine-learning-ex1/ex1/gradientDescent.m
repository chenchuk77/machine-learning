function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
    %GRADIENTDESCENT Performs gradient descent to learn theta
    %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
    %   taking num_iters gradient steps with learning rate alpha

    % Initialize some useful values
    m = length(y); % number of training examples
    J_history = zeros(num_iters, 1);

    for iter = 1:num_iters

        % ====================== YOUR CODE HERE ======================
        % Instructions: Perform a single gradient step on the parameter vector
        %               theta. 
        %
        % Hint: While debugging, it can be useful to print out the values
        %       of the cost function (computeCost) and gradient here.
        %
      
        % work with temp ensures simultanious updates
        % fprintf('\niter = %f\n', iter);
        
        temp = zeros(2, 1);
        
        for i = 1:m
            % computing the term ( the sigma ).
            % the alpha/m will be multiplied after the for loop 
            temp(1) = temp(1) + ((theta(1) + theta(2) * X(:,2)(i)) - y(i));
            temp(2) = temp(2) + ((theta(1) + theta(2) * X(:,2)(i)) - y(i)) * X(:,2)(i);
        end
        
        % multiplying by the constant and update theta simultaniously
        theta(1) = theta(1) - (alpha/m) * temp(1);
        theta(2) = theta(2) - (alpha/m) * temp(2);
        % ============================================================
         
        % Save the cost J in every iteration    
        J_history(iter) = computeCost(X, y, theta);

    end
end
