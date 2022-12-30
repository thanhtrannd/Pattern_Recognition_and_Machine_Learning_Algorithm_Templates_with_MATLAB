load cancer_dataset
X = cancerInputs';
Y = cancerTargets';

[n,d] = size(X);

% Partition for crossvalidation
partition_rate = 0.2;
cv = cvpartition(n, "HoldOut", partition_rate);
Xtest_idx = test(cv);
Xtrain_idx = training(cv);

Xtest = X(Xtest_idx,:);
Ytest_true = Y(Xtest_idx,:);
Xtrain = X(Xtrain_idx,:);
Ytrain_true = Y(Xtrain_idx,:);


[W, Ytest_pred, accuracy] = mlp_train(Xtrain, Ytrain_true, [10 10 2], ["tanh", "sigmoid", "softmax"], "cross-entropy", 100000, 0.00001, Xtest, Ytest_true);

%TODO: ADD REGULARIZATION AND OVERFITTING DETECTION

function [W, Ytest_pred_class, accuracy] = mlp_train(X, Ytrue, network, activationfunctions, lossfunction, maxEpochs, learningrate, Xtest, Ytest)
    % X: input data with rows corresponding to the number of observations and
    % columns corresponding to the number of dimensions

    % Ytrue: True class of X in the format of onehotcode or it will be
    % transformed to that format if not yet

    % network: Vector defining the structure of the network with the number
    % of elements as the number of layers (last layer is the outputlayer,
    % the others are hidden layers), the value of elements defining the
    % number of neurons in the corresponding layer.

    % activationfunctions: Vector of names of activation for each layer, if
    % there is only one name, it will be default activation function for
    % all layers

    [n,d] = size(X);        % n: number of observations, d: number of dimensions
    
    [nY,nClasses] = size(Ytrue);    % nY: number of observations, nClasses: number of classes
    if nClasses == 1
        Ytrue = dummyvar(Ytrue);
        [nY,nClasses] = size(Ytrue);
    end

    % Partition for crossvalidation
    partition_rate = 0.2;
    cv = cvpartition(n, "HoldOut", partition_rate);
    Xval_idx = test(cv);
    Xtrain_idx = training(cv);

    Xval = X(Xval_idx,:);
    Yval_true = Ytrue(Xval_idx,:);
    Xtrain = X(Xtrain_idx,:);
    Ytrain_true = Ytrue(Xtrain_idx,:);
    
    % Extend X and transpose so that rows corresponding to the number of
    % dimensions and columns corresponding to the number of observations
    [nVal, dVal] = size(Xval);
    [nTrain, dTrain] = size(Xtrain);

    Xval_extended = [Xval ones(nVal,1)]';
    Xtrain_extended = [Xtrain ones(nTrain,1)]';
    
    Ytrain_true = Ytrain_true';
    Yval_true = Yval_true';

    [dVal_extended, nVal] = size(Xval_extended);
    [dTrain_extended, nTrain] = size(Xtrain_extended);

    % Initialize network
    nLayers = length(network);          % number of layers in the network
    W = cell(1,nLayers);                % cell array storing weights of layers
    V = cell(1,nLayers);
    Y = cell(1,nLayers);

    V_val = cell(1,nLayers);
    Y_val = cell(1,nLayers);

    for l = 1:nLayers
        if l == 1               % First hidden layer
            W{l} = (rand(dTrain_extended, network(l))-0.5)/10;   % rows as the number of nodes in the layer and the columns as the number of nodes in the previous layer
        elseif l == nLayers     % Output layer
            W{l} = (rand(network(l-1)+1, nClasses)-0.5)/10;
        else                    % Other hidden layers
            W{l} = (rand(network(l-1)+1, network(l))-0.5)/10;
        end
    end
    
    % Define cell array containing handles of activation functions
    % corresponding to each layers
    if ~exist('activationfunctions', 'var')
        activationfunctions = [repmat("", 1, length(network)-1) "softmax"];         % default
    end

    act_functions = cell(1, nLayers);
    valid_act_functions = ["tanh", "sigmoid", "relu", "softmax", ""];
    valid_act_functions_handles = {@my_tanh, @my_sigmoid, @my_relu, @my_softmax, @(x) x};
    
    [valid_activationfunctions, valid_act_functions_idx] = ismember(activationfunctions, valid_act_functions);
    if sum(valid_activationfunctions) ~= length(activationfunctions)
        msg = "Activation function(s) is not valid or not supported!";
        error(msg);
    end
    
    for l = 1:nLayers
        act_functions{l} = valid_act_functions_handles{valid_act_functions_idx(l)};
    end
    
    % Define lossfunction
    if ~exist('lossfunction', 'var')
        lossfunction = "cross-entropy";
    end
    
    valid_loss_functions = ["cross-entropy", "mse", ""];
    valid_loss_functions_handles = {@my_cross_entropy, @my_mean_square_error, @(x) x};
    [valid_lossfunction, valid_loss_functions_idx] = ismember(lossfunction, valid_loss_functions);
    if ~valid_lossfunction
        msg = "Loss function(s) is not valid or not supported!";
        error(msg);
    end
    
    loss_function = valid_loss_functions_handles{valid_loss_functions_idx};

    lossTrain = zeros(1,maxEpochs);
    lossVal = zeros(1,maxEpochs);

    % Initialize other hyperparameters
    if ~exist('maxEpochs', 'var')
        maxEpochs = 10^5;
    end
    
    if ~exist('learningrate', 'var')
        learningrate = 10^-4;
    end
    
    t = 0;
    fig = figure();

    % Start training
    while t < maxEpochs
        t = t +1;

        % Feedforward on training set
        for l = 1:nLayers
            if l == 1               % First hidden layer
                V{l} = W{l}' * Xtrain_extended;   % rows as the number of nodes in the layer and the columns as the number of observations in the training set

            else                    % Other hidden layers and output layers
                V{l} = W{l}' * Y{l-1};   % rows as the number of nodes in the layer and the columns as the number of observations in the training set
            end

            % Apply activation function
            act_func = act_functions{l};
            Y{l} = act_func(V{l});
            % Extend this layer if not output layer
            if l ~= nLayers
                Y{l} = [Y{l}; ones(1,nTrain)];
            end

        end
        
        Ytrain_pred = Y{l};
        
        % Feedforward on validation set
        [Yval_pred_dummy, Yval_pred_class] = feed_forward_net(Xval, W, activationfunctions);

        % Compute loss
        lossTrain(t) = loss_function(Ytrain_pred, Ytrain_true);
        lossVal(t) = loss_function(Yval_pred_dummy, Yval_true);
        
        if (mod(t, 1000) == 0) 
            semilogy(1:t, lossTrain(1:t), "b-"), hold on;
            semilogy(1:t, lossVal(1:t), "r-");
            title(sprintf('Training (epoch %d)', t));
            ylabel('Error');
            legend("training","validation");
            drawnow;
        end

        if lossTrain(t) < 1e-12     % error very small
            break;
        elseif t > maxEpochs        % max number of epochs reached
            break;
        elseif t > 1 && abs(lossTrain(t)-lossTrain(t-1)) < 1e-12    % if not improve
            break
        end
        
        % Backpropagation
        delta = cell(size(W));
        delta_W = cell(size(W));

        for l = nLayers:-1:1
            if l == nLayers         % Output layer
                delta{l} = Ytrain_pred - Ytrain_true;
            else                    % Hidden layers
                delta{l} = W{l+1}(1:end-1,:) * delta{l+1} .* (1-Y{l}(1:end-1,:) .^2);
            end
            
            if l == 1
                delta_W{l} = -learningrate * Xtrain_extended * delta{l}';
            else
                delta_W{l} = -learningrate * Y{l-1} * delta{l}';
            end
            W{l} = W{l} + delta_W{l};
        end

    end
    
    % Testing
    Ytest_pred = NaN;
    if exist("Xtest", 'var')
        [Ytest_pred, Ytest_pred_class] = feed_forward_net(Xtest, W, activationfunctions);
    end
    
    accurary = NaN;
    if exist('Ytest', 'var')
        [nY,nClasses] = size(Ytest);    % nY: number of observations, nClasses: number of classes
        if nClasses ~= 1
            [~,Ytest] = max(Ytest, [], 2);
            [nY,nClasses] = size(Ytest);
        end
        accuracy = mean(Ytest_pred_class == Ytest');
    end

end

function Y = my_tanh(X)         % tanh(x) = (exp(x)-exp(-x))/(exp(x)+exp(-x))
    Y = (exp(X) - exp(-X)) ./ (exp(X) + exp(-X));
end

function Y = my_sigmoid(X)      % logistics(x) = 1/(1+exp(-x))
    Y = 1 ./ (1 + exp(-X));
end

function Y = my_relu(X)         % rectifier(x) = max (0, x)
    Y = max([zeros(size(X)) X], [], 2);
end

function Y = my_softmax(X)      % softmax(x) = exp(x)/sum(exp(x))
    Y = exp(X)./sum(exp(X));
end

function loss = my_cross_entropy(Ypred, Ytrue)
    loss = mean(-sum(Ytrue.*log(Ypred)));
end

function loss = my_mean_square_error(Ypred, Ytrue)
    loss = mean(sum((Ypred-Ytrue).^2));
end

function [Ypred_dummy, Ypred_class] = feed_forward_net(X, W, activationfunctions)
    [n,d] = size(X);
    X_extended = [X ones(n,1)]';

    V = cell(size(W));
    Y = cell(size(W));
    
    nLayers = length(W);

    % Define cell array containing handles of activation functions
    % corresponding to each layers
    if ~exist('activationfunctions', 'var')
        activationfunctions = [repmat("", 1, length(network)-1) "softmax"];         % default
    end

    act_functions = cell(1, nLayers);
    valid_act_functions = ["tanh", "sigmoid", "relu", "softmax", ""];
    valid_act_functions_handles = {@my_tanh, @my_sigmoid, @my_relu, @my_softmax, @(x) x};
    
    [valid_activationfunctions, valid_act_functions_idx] = ismember(activationfunctions, valid_act_functions);
    if sum(valid_activationfunctions) ~= length(activationfunctions)
        msg = "Activation function(s) is not valid or not supported!";
        error(msg);
    end
    
    for l = 1:nLayers
        act_functions{l} = valid_act_functions_handles{valid_act_functions_idx};
    end

    % Feed forward
    for l = 1:nLayers
        if l == 1               % First hidden layer
            V{l} = W{l}' * X_extended;   % rows as the number of nodes in the layer and the columns as the number of observations in the training set
        else                    % Other hidden layers and output layers
            V{l} = W{l}' * Y{l-1};   % rows as the number of nodes in the layer and the columns as the number of observations in the training set
        end
        
        % Apply activation function
        act_func = act_functions{l};
        Y{l} = act_func(V{l});

        % Extend this layer if not output layer
        if l ~= nLayers
            Y{l} = [Y{l}; ones(1,n)];
        end
    end

    Ypred_dummy = Y{nLayers};
    [~,Ypred_class] = max(Ypred_dummy);

end
