"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling by max.
 - Digit classification is implemented with a logistic regression rather than an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner: Gradient-Based Learning Applied to Document Recognition, 
   Proceedings of the IEEE, 86(11):2278-2324, 1998. http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import logging
logging.getLogger("theano.gof.cmodule").setLevel(logging.DEBUG)

#####################      LOAD LOGISTIC REGRESSION LAYER FROM logistic_sgd.py
######################################################################
######################################################################
from logistic_sgd import LogisticRegression, load_data
#####################      HIDDEN LAYER FROM multi_layer_perceptron.py
######################################################################
######################################################################
from multi_layer_perceptron import HiddenLayer

####################           LeNet CONVOLUTION POOL LAYER CLASS
######################################################################
######################################################################
######################################################################
######################################################################
class LeNetConvPoolLayer(object):
    """Convolutional then pool layer (of a  network)"""

    def __init__(self, rng, input_data, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Model initialization: W, b, input_data, output

        rng: numpy.random.RandomState: a random number generator used to initialize weights
        input_data: theano.tensor.dtensor4: symbolic image tensor, of shape image_shape
        filter_shape: tuple or list of length 4: (number of filters, num input feature maps, filter height, filter width)
        image_shape: tuple or list of length 4: (batch size, num input feature maps, image height, image width)
        poolsize: tuple or list of length 2: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input_data = input_data

        # there are "num input feature maps * filter height * filter width"  input_datas to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from: "num output feature maps * filter height * filter width" / pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) / numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX  # @UndefinedVariable
            ),
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)  # @UndefinedVariable
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input_data,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )

        # add the bias term. Since the bias is a vector (1D array), we first reshape it to a tensor of shape (1, n_filters, 1, 1). 
        # Each bias will thus be broadcasted across mini-batches and feature map width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.1, n_epochs=200,
                    dataset='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=500):
    """ Demonstrates lenet on MNIST dataset

    learning_rate: float: learning rate used (factor for the stochastic gradient)
    n_epochs: int: maximal number of epochs to run the optimizer
    dataset: string: path to the dataset used for training /testing (MNIST here)
    nkerns: list of ints: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    ############################ CREATE SYMBOLIC VARIABLES
    ##################################################################
    ##################################################################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    # Reshape matrix of rasterized input images of shape (batch_size, 28 * 28) to a 4D tensor (compatible with our LeNetConvPoolLayer)
    # (28, 28) is the size of MNIST images.
    layer0_input_data = x.reshape((batch_size, 1, 28, 28))

    # Construct the first convolutional pooling layer:
        # filtering reduces the image size to (28-5+1 , 28-5+1) = (24, 24)
        # maxpooling reduces this further to (24/2, 24/2) = (12, 12)
        # 4D output tensor is thus of shape (batch_size, nkerns[0], 12, 12)
    layer0 = LeNetConvPoolLayer(
        rng,
        input_data=layer0_input_data,
        image_shape=(batch_size, 1, 28, 28),
        filter_shape=(nkerns[0], 1, 5, 5),
        poolsize=(2, 2)
    )

    # Construct the second convolutional pooling layer:
        # filtering reduces the image size to (12-5+1, 12-5+1) = (8, 8)
        # maxpooling reduces this further to (8/2, 8/2) = (4, 4)
        # 4D output tensor is thus of shape (batch_size, nkerns[1], 4, 4)
    layer1 = LeNetConvPoolLayer(
        rng,
        input_data=layer0.output,
        image_shape=(batch_size, nkerns[0], 12, 12),
        filter_shape=(nkerns[1], nkerns[0], 5, 5),
        poolsize=(2, 2)
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 4 * 4), or (500, 50 * 4 * 4) = (500, 800) with the default values.
    #### batch will be taken care of by the input of the function we don't need to worry about it. 
    #### n_out = 500 is just chosen for no reason and it only needs to be equal to the next layer input. no other reason.
    layer2_input_data = layer1.output.flatten(2)

    # construct a fully-connected layer (tanh)
    layer2 = HiddenLayer(
        rng,
        input_data=layer2_input_data,
        n_in=nkerns[1] * 4 * 4,
        n_out=500,
        activation=T.tanh
    )

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input_data=layer2.output_data, n_in=500, n_out=10)

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y)

    # create a function to compute the mistakes that are made by the model
    test_model_error = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        [index],
        layer3.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by SGD Since this model has many parameters, 
    # it would be tedious to manually create an update rule for each model parameter. 
    # We thus create the updates list by automatically looping over all (params[i], grads[i]) pairs.
    updates = [
        (param_i, param_i - learning_rate * grad_i)
        for param_i, grad_i in zip(params, grads)
    ]

    train_model = theano.function(   # THIS IS WHERE SEGMENTATION FAULT 11 HAPPENS
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ####################################      TRAIN MODEL
    ##################################################################
    ##################################################################
    print '... training'
    
    # early-stopping parameters
    
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    # go through this many minibatche before checking the network on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)


    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    
    #=============      VISUALIZATION
    error_list = []
    #=============      VISUALIZATION

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            train_model(minibatch_index)
            iter_num = (epoch - 1) * n_train_batches + minibatch_index
            
            if iter_num % 100 == 0:
                print 'training @ iter = ', iter_num

            if (iter_num + 1) % validation_frequency == 0:

                # VALIDATE
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter_num

                    # test it on the test set
                    test_losses = [test_model_error(i) for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print('     epoch %i, minibatch %i/%i, test error of best model UPDATED %f %%' % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))


            if patience <= iter_num:
                done_looping = True
                break

    print "LOOP FINISHED\n"         
    end_time = time.clock()
    print('Optimization complete. Best validation score of %f %% obtained at iteration %i, with test performance %f %%' % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print >> sys.stderr, ('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))

    # ==========================================  ERROR VISUALIZATION  
    import numpy as nx
    import pylab as p
    
    x = nx.arange(0, epoch)
    y = error_list 
    
    p.plot(x,y, color='red', lw=2)
    p.show()
    
    layer3.visualizeWeights()
    # ==========================================  ERROR VISUALIZATION

if __name__ == '__main__':
    evaluate_lenet5()


def experiment(state, channel):
    evaluate_lenet5(state.learning_rate, dataset=state.dataset)
