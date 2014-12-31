function [embedding_layer_state, hidden_layer_state, output_layer_state] = ...
  fprop(input_batch, word_embedding_weights, embed_to_hid_weights,...
  hid_to_output_weights, hid_bias, output_bias)
% This method forward propagates through a neural network.
% Note: first hidden layer (Word Embedding) is hidden1 and the second hidden
% layer (Hidden Layer) is hidden2
%
% We have a 4-gram from the first 3-grams we want to know what is the 4th
% gram. 
%
% Inputs:
%   input_batch: (mini batch of 3 grams) The input data as a numwords X batchsize matrix:
%     numwords   : the number of words (In this case 3 as in 3 gram)
%     batchsize  : the number of data points.
%   So, if input_batch(i, j) = k then the ith word in data point j is word
%     index k of the vocabulary.
%
%   word_embedding_weights: Word embedding weights as a vocab_size X numhid1
%   matrix:
%     vocab_size : the size of the vocabulary (250 words in this example)
%     numhid1    : dimensionality of the embedding space. Dimensionality of embedding space; default = 50.
%
%   embed_to_hid_weights: Weights between the word embedding layer and
%   hidden (from 3*50 neurons in embed layer to 200 neurons in the hidden layer)
%   layer as a numhid1*numwords X numhid2 matrix: 
%     numhid2    :   number of hidden units. Number of units in hidden layer; default = 200.
%
%   hid_to_output_weights: Weights between the hidden layer and output softmax
%   unit as a numhid2 X vocab_size matrix:
%
%   hid_bias: Bias of the hidden layer as a numhid2 X 1 matrix.
%
%   output_bias: Bias of the output layer as a matrix of size vocab_size X 1.
%
% Outputs:
%   embedding_layer_state: State of units in the embedding layer as a matrix of
%     size numhid1*numwords X batchsize
%
%   hidden_layer_state: State of units in the hidden layer as a matrix of size
%     numhid2 X batchsize
%
%   output_layer_state: State of units in the output layer as a matrix of size
%     vocab_size X batchsize
%

[numgrams, batchsize] = size(input_batch);
[vocab_size, numhid1] = size(word_embedding_weights); % vocabulary embedding weights
numhid2 = size(embed_to_hid_weights, 2);

%% COMPUTE STATE OF WORD EMBEDDING LAYER.
    
    % stack the columns together one after another, one 3-gram at a time. like reshape([1 2 3 4; 7 8 9 0 ],1,[])
    input_batch_reshaped = reshape(input_batch, 1, []); 
   
    % extract weights of each of the vocabulary instances in the input. one parameter set
    % for each of the 300 words for each of the 50 neurons in Embedding layer, hence a 300*50
    word_embedding_weights_lookedup = word_embedding_weights(input_batch_reshaped,:)';

    % reshape 300*50 to 3*50*100 so all 3gram parameters in Embed layer are
    % in line with  each other, and we have 100 minibatches
    embedding_layer_state = reshape(word_embedding_weights_lookedup, numhid1 * numgrams, []);

%% COMPUTE STATE OF HIDDEN LAYER.
% Compute inputs to hidden units: weight*state+bias
% each underlying neuron times its weight + bias. TODO bias is zero for embeded layer state
repmat_of_bias_for_minibatch = repmat(hid_bias, 1, batchsize);
inputs_to_hidden_units = embed_to_hid_weights' * embedding_layer_state + repmat_of_bias_for_minibatch; 

% Apply logistic activation function.
% FILL IN CODE. Replace the line below by one of the options.
hidden_layer_state = zeros(numhid2, batchsize);
% Options
% (a) hidden_layer_state = 1 ./ (1 + exp(inputs_to_hidden_units));
% (b) hidden_layer_state = 1 ./ (1 - exp(-inputs_to_hidden_units));
% (c) hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hidden_units));
hidden_layer_state = 1 ./ (1 + exp(-inputs_to_hidden_units));
% (d) hidden_layer_state = -1 ./ (1 + exp(-inputs_to_hidden_units));

%% COMPUTE STATE OF OUTPUT LAYER.
% Compute inputs to softmax.
% FILL IN CODE. Replace the line below by one of the options.
inputs_to_softmax = zeros(vocab_size, batchsize);
% Options
% (a) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, 1, batchsize);
inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, 1, batchsize);
% (b) inputs_to_softmax = hid_to_output_weights' * hidden_layer_state +  repmat(output_bias, batchsize, 1);
% (c) inputs_to_softmax = hidden_layer_state * hid_to_output_weights' +  repmat(output_bias, 1, batchsize);
% (d) inputs_to_softmax = hid_to_output_weights * hidden_layer_state +  repmat(output_bias, batchsize, 1);

% Subtract maximum. 
% Remember that adding or subtracting the same constant from each input to a
% softmax unit does not affect the outputs. Here we are subtracting maximum to
% make all inputs <= 0. This prevents overflows when computing their
% exponents.
inputs_to_softmax = inputs_to_softmax...
  - repmat(max(inputs_to_softmax), vocab_size, 1);

% Compute exp.
output_layer_state = exp(inputs_to_softmax);

% Normalize to get probability distribution.
output_layer_state = output_layer_state ./ repmat(...
  sum(output_layer_state, 1), vocab_size, 1);
