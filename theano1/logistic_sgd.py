"""  THIS IS ESSENTIALLY A SINGLE-LAYER LINEAR NETWORK WITH SOFTMAX AS OUTPUT loss is zero-one loss
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix `W` and a bias vector `b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as a single layer neural network 
with its final layer as softmax layer:

  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
                &= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

  y_{pred} = argmax_i P(Y=i|x,W,b)

This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.

References:

    - textbooks: "Pattern Recognition and Machine Learning" -
                 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T


class LogisticRegression(object):
    """Multi-class Logistic Regression Class : Here logistic regression is 
     described with a weight matrix `W` and bias vector `b`. 
    Classification is done by projecting data points onto a set of hyperplanes, 
    the distance to which is used to determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
        """ Init

        input:      theano.tensor.TensorType: symbolic variable that describes 
                    the input of the architecture (one minibatch)

        n_in: int:  number of input units, the dimension of the space in
                    which the datapoints lie

        n_out: int: number of output units, the dimension of the space in
                    which the labels lie
        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )

        # symbolic theano expression for computing the matrix of class-membership
        # probabilities:
        # W is a matrix where column-k represent the separation hyper plain for class-k
        # x is a matrix where row-j represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper plain-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic theano expression to determine class with maximum probability
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """ Return the mean of the negative log-likelihood of the prediction

        y: theano.tensor.TensorType: a vector of correct labels for each example

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0]: (symbolically) the number of rows in y, i.e., number of examples (n) in minibatch
        # T.arange(y.shape[0]): a symbolic vector which will contain [0,1,2,... n-1] 
        # T.log(self.p_y_given_x): a matrix of Log-Probabilities (LP) with one row per example and one column per class
        # LP[T.arange(y.shape[0]),y]: a vector v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ..., LP[n-1,y[n-1]]] 
        # T.mean(LP[T.arange(y.shape[0]),y]): the mean (across minibatch examples) of the elements in v, i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return the average number of errors in the minibatch (zero-one loss)

        y: theano.tensor.TensorType: a vector of correct labels for each example
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.y_pred.type)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y)) # RETURN AVERAGE OF ERRORS
        else:
            raise NotImplementedError()

# =============     END CLASS DEFINITION 
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################










def load_data(dataset):
    ''' Loads the dataset (train, validation and test) into shared variables

    dataset: string: the path to the dataset (here MNIST)
    '''

    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    # train_set, valid_set, test_set : tuple(input, target)
    # input is an numpy.ndarray of images (2d matrix) each row correspond to an example. 
    # target is a numpy.ndarray a vector of target values for each image.

    def shared_dataset(data_xy, borrow=True):
        """ load the dataset into shared variables

        Shared variables allow Theano to copy data into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime would lead 
        to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
       
        # When storing data on the GPU it has to be stored as floats (floatX). 
        # But during our computations we need labels as ints (we use labels as index, and if they are
        # floats it doesn't make sense). Therefore the cast y's to int. 
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


##################################################################
##################################################################
##################################################################
##################################################################
##################################################################








# The actual parameter optimization
def sgd_optimization_mnist(learning_rate=0.13, 
                           n_epochs=1000,
                           dataset='mnist.pkl.gz',
                           batch_size=600):
    """
    stochastic gradient descent optimization of our model (log-linear model)
    """
    
    print '... loading data'
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # set number of minibatches for each set
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... building symbolic model'

    ############################ CREATE SYMBOLIC VARIABLES
    ##################################################################
    ##################################################################

    # create symbolic variable for the indexing minibatches
    index = T.lscalar()  # index to a [mini]batch

    # create symbolic variables for input (x and y represent a minibatch)
    x = T.matrix('x')  # data image
    y = T.ivector('y')  # labels, presented as 1D vector of [int] labels

    # initialize the logistic regression class (w and b). Each image is 28*28
    classifier = LogisticRegression(input=x, n_in=28 * 28, n_out=10)

    # Symbolic format of the cost function we minimize during training: 
    # is the negative log likelihood of the model 
    cost = classifier.negative_log_likelihood(y)

    # Compiled Theano function for model errors in a minibatch
    test_model_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    validation_model_error = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to each parameter
    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    # compiling a Theano function `train_model` that returns the cost, but ALSO
    # update the parameter of the model based on the rules defined in `updates` (variable, update expression) pairs
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=[(classifier.W, classifier.W - learning_rate * g_W),
                 (classifier.b, classifier.b - learning_rate * g_b)],
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


    ####################################      TRAIN MODEL
    ##################################################################
    ##################################################################

    print '... training the model'
   
    # early-stopping parameters
    
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    # go through this many minibatche before checking the network on the validation set; in this case we check every epoch
    validation_frequency = min(n_train_batches, patience / 2)
                                  

    best_validation_loss = numpy.inf
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

            train_model(minibatch_index) # discard train minibatch_avg_cost

            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
               
                # VALIDATE
                # compute zero-one loss on validation set
                validation_losses = [validation_model_error(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                   
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
    print('Optimization complete with best validation score of %f %%, with test performance %f %%' % (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('Total train/validation time of ' + os.path.split(__file__)[1] + ' ran for %.1fs' % ((end_time - start_time)))
   
    # ==========================================  ERROR VISUALIZATION
    
    import numpy as nx
    import pylab as p
    
    x = nx.arange(0, epoch)
    y = error_list 
    
    p.plot(x,y, color='red', lw=2)
    p.show()
    
    ########### VISUALIZE WEIGHTS
    
    import matplotlib as mpl
    from matplotlib import pyplot
    
    fig = pyplot.figure(1)
    cmap = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', ['black','white'], 256)
    
    zvals = classifier.W.get_value()
    zvals = zvals.reshape(28,28,10)

    ########### VISUALIZE WEIGHTS 
    
    a1 = zvals[:,:,1].reshape(28,28)
    pyplot.imshow(a1, interpolation='nearest', cmap = cmap, origin='lower')
    fig.canvas.draw()
    
    fig = pyplot.figure(2)
    a1 = zvals[:,:,2].reshape(28,28)
    pyplot.imshow(a1, interpolation='nearest', cmap = cmap, origin='lower')
    fig.canvas.draw()

    fig = pyplot.figure(3)
    a1 = zvals[:,:,3].reshape(28,28)
    pyplot.imshow(a1, interpolation='nearest', cmap = cmap, origin='lower')
    fig.canvas.draw()

    fig = pyplot.figure(4)
    a1 = zvals[:,:,4].reshape(28,28)
    pyplot.imshow(a1, interpolation='nearest', cmap = cmap, origin='lower')
    fig.canvas.draw()

    fig = pyplot.figure(5)
    a1 = zvals[:,:,5].reshape(28,28)
    pyplot.imshow(a1, interpolation='nearest', cmap = cmap, origin='lower')
    fig.canvas.draw()     

    pyplot.show()
    ########### VISUALIZE WEIGHTS
    # ==========================================  ERROR VISUALIZATION

    

    
    

if __name__ == '__main__':
    print "Hi!"
    sgd_optimization_mnist()
