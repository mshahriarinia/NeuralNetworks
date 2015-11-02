'''
   TODO lasagne seed initializetion
   TODO calculate hwo many layers/parameters for network and how layers are connected
   (c) Morteza Shahriari Nia (http://mshahriarinia.com)
'''
from __future__ import print_function

import sys
import os
import time

#from sklearn.datasets import load_svmlight_file

import numpy as np
import theano
#import theano.tensor as T

from theano import tensor as T, function, printing

import lasagne

from pprint import pprint
import scipy.sparse as sps

# ################## Download the dataset ##################

def load_dataset():
    def svm_read_problem(data_file_name, dim):
        """
        svm_read_problem(data_file_name) -> [y, x]
        Read LIBSVM-format data from data_file_name and return labels y
        and data instances x.
        """
        prob_y = []
        prob_x = []
        for line in open(data_file_name):
            line = line.split(None, 1)
            # In case an instance with all zero features
            if len(line) == 1: line += ['']
            label, features = line
            xi = {}
            for e in features.split():
                ind, val = e.split(":")
                xi[int(ind)] = float(val)
            #prob_y += [int(label)-1] 
            # The problem was that libsvm labels were provided in {0,1}, however in standard form it should have been {1,2} 
            # (hence the assumption in all libsvm loaders and subtracting class labels by one). By adjusting that and not subtracting 
            # from one, it worked!
            if int(label) == 1:
                prob_y += [1]
            elif int(label) == -1:
                prob_y += [2]
            prob_x += [xi]
        input = np.zeros([len(prob_x),dim])
        output = np.array(prob_y)
        for i in range(len(prob_x)):
            for inx in prob_x[i]:
              input[i][inx-1] = prob_x[i][inx]
        return (np.float32(input), output.astype(np.int32))
   
    start_time = time.time()

    # libsvm format si sparse, here we convert to dense format. data should be of float type and labels of int32
    #X_train, y_train = svm_read_problem("/data1/shahriari/train.txt", 728)
    #X_test, y_test = svm_read_problem("/data1/shahriari/test.txt", 728)
    DATA_PATH = "/data1/chori/work/DriverStatus/open/data4libsvm/w1/data/marge_150304_003_07101409/171/"
    X_train, y_train = svm_read_problem(DATA_PATH+"train_marge_150304_003_07101409.171_5-165,167-170,172-183_0.libsvm", 728)
    X_test, y_test = svm_read_problem(DATA_PATH + "test_marge_150304_003_07101409.171_5-165,167-170,172-183_0.libsvm", 728)

    # Take the last 10000 training examples for validation. TODO
    # take 10% instead

    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]

    print("Loading the dataset took {:.3f}s".format( time.time() - start_time))

    return X_train, y_train, X_val, y_val, X_test, y_test


# ################## Get minibatches ##########################
# This is just a simple helper function iterating over training data in mini-batches of a particular size, optionally in random order. It assumes data is available 
# as numpy arrays. For big datasets, you could load numpy arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your own custom data iteration function. 
# For small datasets, you can also copy them to GPU at once for slightly improved performance. This is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# We could add some weight decay as well here, checkout lasagne.regularization.

def main(num_epochs=50, minibatch_size=2048):  # TODO set minibatch higher e.g. 500
    
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    train_batches = X_train.shape[0]/minibatch_size
    val_batches = X_val.shape[0]/minibatch_size
 
    pprint(X_train)
    
    # Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    # ################### network
    print("Building model and compiling functions...")
    # Input layer, specifying the expected input shape of the network (batchsize 256, 728 channela) and linking it to the given Theano variable `input_var`, if any:
    
    l_in = lasagne.layers.InputLayer(shape=(None, 728), input_var=input_var)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

    l_hid1 = lasagne.layers.DenseLayer(l_in_drop, num_units=256, nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotUniform())
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop, num_units=64, nonlinearity=lasagne.nonlinearities.sigmoid, W=lasagne.init.GlorotUniform())
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

    l_out = lasagne.layers.DenseLayer(l_hid2_drop, num_units=2, nonlinearity=lasagne.nonlinearities.softmax)

    network = l_out

    # #################### Loss expressions
    # Loss expression for training, i.e., a scalar objective we want to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    train_loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    train_loss = train_loss.mean()  

    # Loss expression for validation/test. Difference: We do a deterministic forward pass (deterministic=True) through the network, disabling dropout layers
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction, target_var)
    test_loss = test_loss.mean()

    # #################### Test accuracy
    # Expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX) # TODO axis=1

    # Create update expressions for training. Here, we'll use Stochastic Gradient Descent (SGD) with Nesterov momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(train_loss, params, learning_rate=0.0001, momentum=0.9)


    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], train_loss, updates=updates)
    # Compile a function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # ############################### Debugging output capabilities
    layers = lasagne.layers.get_all_layers(network)
    activations = lasagne.layers.get_output(layers)
    print(activations) 
    # [inputs, Elemwise{mul,no_inplace}.0, sigmoid.0, Elemwise{mul,no_inplace}.0, sigmoid.0, Elemwise{mul,no_inplace}.0, Softmax.0]
    activation_fn= theano.function([input_var], activations[6]
            )
   

    #test_acc0 = T.eq(T.argmax(test_prediction, axis=1), target_var)
    #test_acc1 = T.argmax(test_prediction, axis=1)

   # test_acc0_fn =  theano.function([input_var, target_var], test_acc0)
   # test_acc1_fn =  theano.function([input_var], test_acc1)

    
    # ####

    print("Starting training...")
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Full pass over the training data:
        train_err = 0
        for batch in iterate_minibatches(X_train, y_train, minibatch_size, shuffle=True):            
            inputs, targets = batch
            train_err += train_fn(inputs, targets)

        # Full pass over the validation data:
        val_err = 0
        val_acc = 0
        for batch in iterate_minibatches(X_val, y_val, minibatch_size, shuffle=False):
            inputs, targets = batch
           # print( activation_fn(inputs))
           # print(test_acc0_fn([inputs,targets]))
           # print(test_acc1_fn(inputs))
            

            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc

        print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))


    # ######################### Test set results
    # At the end of training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, minibatch_size, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)

main()

