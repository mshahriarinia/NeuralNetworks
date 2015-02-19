"""
This tutorial introduces the multilayer perceptron using Theano.

 A multilayer perceptron is a logistic regressor where instead of feeding the input to the logistic regression,
 you insert a intermediate layer, called the hidden layer, that has a nonlinear activation function 
 (usually tanh or sigmoid). One can use many such hidden layers making the architecture deep. 
 The tutorial will also tackle the problem of MNIST digit classification.

    f(x) = G( b^{(2)} + W^{(2)}( s( b^{(1)} + W^{(1)} x))),

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 5
"""
__docformat__ = 'restructedtext en'


import os
import sys
import time

import numpy

import theano
import theano.tensor as T

#####################      LOAD LOGISTIC REGRESSION LAYER FROM logistic_sgd.py
######################################################################
######################################################################
from logistic_sgd import LogisticRegression, load_data

####################           HIDDEN LAYER CLASS
######################################################################
######################################################################
######################################################################
######################################################################
class HiddenLayer(object):
    """ Typical hidden layer of a MLP: units fully-connected and have sigmoid/tanh activation function. 
        Weight is (n_in,n_out) matrix and bias is (n_out,) vector. each node's activation: tanh(dot(input,W) + b)
    """
    def __init__(self, rng, input_data, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        rng: numpy.random.RandomState: a random number generator used to initialize weights
        input_data: theano.tensor.dmatrix: a symbolic tensor of shape (n_examples, n_in)
        n_in: int: dimensionality of input
        n_out: int: number of hidden units
        activation: theano.Op or function: Non linearity to be applied in the hidden layer. Here we apply `tanh`
        """
        self.input_data = input_data

        # `W` is uniformely sampled, The output is converted to theano.config.floatX so that code is runable on GPU
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low  = -numpy.sqrt(6. / (n_in + n_out)),
                    high =  numpy.sqrt(6. / (n_in + n_out)),
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX  # @UndefinedVariable
            )
            
            # [Xavier10] suggests to use 4 times larger initial weights for sigmoid compared to tanh
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  # @UndefinedVariable
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input_data, self.W) + self.b
        self.output_data = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]


# =============     END CLASS DEFINITION - HiddenLayer
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one or more layers of hidden units with nonlinear activations (sigmoid or tanh).
    here defined by a ``HiddenLayer`` class)  while the top layer is a softamx layer (defined 
    here by a ``LogisticRegression`` class).
    """

    def __init__(self, rng, input_data, n_in, n_hidden, n_out):
        """Initialize the parameters for the multilayer perceptron

        rng: numpy.random.RandomState: a random number generator used to initialize weights
        input_data: theano.tensor.TensorType: symbolic variable that describes the input of the architecture (one minibatch)
        n_in: int: number of input units, the dimension of the space in which the datapoints lie
        n_hidden: int: number of hidden units
        n_out: int: number of output units, the dimension of the space in which the labels lie
        """

        # Since we are dealing with a one hidden layer MLP, this will translate into a HiddenLayer with 
        # a tanh activation function (or other activation functions) connected to the LogisticRegression layer
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input_data=input_data,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        # The logistic regression layer gets as input the hidden units of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input_data=self.hiddenLayer.output_data,
            n_in=n_hidden,
            n_out=n_out
        )
        # L1 norm ; one regularization option is to enforce L1 norm to be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum()
            + abs(self.logRegressionLayer.W).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum()
            + (self.logRegressionLayer.W ** 2).sum()
        )

        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params

# =============     END CLASS DEFINITION - MLP
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

# actual parameter optimization
def mlp_sgd_optimization_mnist(learning_rate=0.01, L1_reg=0.00, L2_reg=0.0001, n_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=20, n_hidden=500):
    """
    Stochastic gradient descent optimization for a multilayer perceptron on MNIST dataset

    learning_rate: float: learning rate used (factor for the stochastic gradient)
    L1_reg: float: L1-norm's weight when added to the cost
    L2_reg: float: L2-norm's weight when added to the cost
    n_epochs: int: maximal number of epochs to run the optimizer
    dataset: string: path of file http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz
   """
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
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input_data=x,
        n_in=28 * 28,
        n_hidden=n_hidden,
        n_out=10
    )

    # the cost we minimize during training is the negative log likelihood of the model plus the regularization terms (L1 and L2); 
    # cost is expressed here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # Symbolic function that computes the mistakes that are made by the model on a minibatch
    test_model_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to each parameter
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of (variable, update expression) pairs
    # >>> x = [1, 2, 3]; y = [4, 5, 6]; zipped = zip(x, y)
    # >>> zipped
    # [(1, 4), (2, 5), (3, 6)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # Symbolic function that returns the cost, and updates the parameter of the model
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
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

    done_looping = False
    epoch = 0

    #=============      VISUALIZATION
    error_list = []
    #=============      VISUALIZATION

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            train_model(minibatch_index)  # discard train minibatch_avg_cost
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                
                # VALIDATE
                # compute zero-one loss on validation set
                validation_losses = [validate_model_error(i) for i
                                     in xrange(n_valid_batches)]
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

        # Keep each epoch's ERROR to visualize them all at the end
        error_list.append(best_validation_loss)
        # =============================                       EPOCH LOOP


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
    # ==========================================  ERROR VISUALIZATION

    

    
    

if __name__ == '__main__':
    mlp_sgd_optimization_mnist()