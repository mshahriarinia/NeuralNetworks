"""
 Denoising Auto-Encoders (DAE) are the building blocks for SDAE. They are based on Bengio et al. 2007. 
 
 Autoencoder (AE): takes an input x and first maps it to a hidden representation 
            y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. 
     The resulting latent representation y is then mapped back to a "reconstructed" vector z \in [0,1]^d in input space 
            z = g_{\theta'}(y) = s(W'y + b').  
     The weight matrix W' can optionally be constrained such that 
            W' = W^T,     the autoencoder is said to have tied weights. 
     The network is trained such that to minimize the reconstruction error (the error between x and z).

 DAE: x is corrupted to \tilde{x}. Afterwards y is computed as before 
             y = s(W\tilde{x} + b) and z as s(W'y + b'). 
     The reconstruction error is now measured between z and the uncorrupted input x, which is computed as the cross-entropy:
          - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]

 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: 
     Extracting and Composing Robust Features with Denoising Autoencoders,    ICML'08, 1096-1103, 2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: 
     Greedy Layer-Wise Training of Deep Networks,    Advances in Neural Information Processing Systems 19, 2007
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image


class DAE(object):
    """Denoising Auto-Encoder (DAE) tries to reconstruct the input from a corrupted version of it by projecting it first in a latent space and 
    reprojecting it afterwards back in the input space. (Refer to Vincent et al. 2008)
    """

    def __init__(self, numpy_rng, theano_rng=None,
                 input=None, n_visible=784, n_hidden=500,
                 W=None, bhid=None, bvis=None):
        """
        numpy_rn  : numpy.random.RandomState: number random generator used to generate weights
        theano_rng: theano.tensor.shared_randomstreams.RandomStreams: Theano random generator; if None is given one is generated based on a seed drawn from `rng`
        input     : theano.tensor.TensorType: a symbolic description of the input or None for standalone DAE
        n_visible : int: number of visible units (the dimension d of the input )
        n_hidden  : int:  number of hidden units (the dimension d' of the latent or hidden space)
        W         : theano.tensor.TensorType: Theano variable pointing to a set of weights that should be shared belong the DAE and another architecture; if DAE should be standalone set this to None
        bhid      : theano.tensor.TensorType: Theano variable pointing to a set of biases values (for hidden units) that should be shared belong DAE and another architecture; if DAE should be standalone set this to None
        bvis      : theano.tensor.TensorType: Theano variable pointing to a set of biases values (for visible units) that should be shared belong DAE and another architecture; if DAE should be standalone set this to None
            
        Symbolic variables are useful when the input is the result of some computations, or when weights are shared between DAE and an MLP layer. 
        With SDAEs the DAE on layer 2 gets as input the output of the DAE on layer 1, and the weights of the DAE are used in the second stage of training to construct an MLP.
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # Note: W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W uniformely sampled, output converted using asarray to theano.config.floatX, so that the code is runnable on GPU
            W_bound = 4 * numpy.sqrt(6. / (n_hidden + n_visible))
            initial_W = numpy.asarray(numpy_rng.uniform(low  =-W_bound, 
                                                        high = W_bound, 
                                                        size=(n_visible, n_hidden)), 
                                      dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(value=numpy.zeros(n_visible, dtype=theano.config.floatX), borrow=True)

        if not bhid:
            bhid = theano.shared(value=numpy.zeros(n_hidden, dtype=theano.config.floatX), name='b', borrow=True)

        self.W = W
        self.b = bhid         # b corresponds to the bias of the hidden
        self.b_prime = bvis           # b_prime corresponds to the bias of the visible
        self.W_prime = self.W.T         # tied weights, therefore W_prime is W transpose

        self.theano_rng = theano_rng
        
        if input is None:
            # if no input is given, generate a variable representing the input.
            # we use a matrix because we expect a minibatch of several examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
        
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

    # theano.rng.binomial: this will produce an array of 0s and 1s where 1 has a  probability of 1 - ``corruption_level`` and 0 with  ``corruption_level``
    # Note: The binomial function return int64 data type by default.  int64 multiplicated by the input type(floatX) always return float64.  
    # To keep all data in floatX when floatX is float32, we set the dtype of the binomial to floatX. Here the value of the binomial is always 0 or 1, this don't change the result. 
    # This is needed to allow the gpu to work correctly as it only support float32 for now.
    # shape(size) of random numbers that it should produce  # number of trials  # probability of success of each trial

    def get_corrupted_input(self, input, corruption_level):
        """zero-out randomly selected subset of size coruption_level"""
        corrupted_index = self.theano_rng.binomial(size=input.shape, n=1, p=1-corruption_level, dtype=theano.config.floatX)
        return corrupted_index * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer"""
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng step of the DAE """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using minibatches, L will be a vector, with one entry per example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        
        # note : L is a vector, each element is the cross-entropy cost of the reconstruction of the corresponding example of the minibatch.
        #        We need to compute the average of all these to get the cost of the minibatch
        cost = T.mean(L)

        g_params = T.grad(cost, self.params)      # compute the gradients of the cost of the `DAE` with respect to its parameters
        updates = [(param, param - learning_rate * gparam)  for param, gparam in zip(self.params, g_params)]   # generate the list of updates

        return (cost, updates)
    
# =============     END CLASS DEFINITION 
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################










def test_DAE(learning_rate=0.1, training_epochs=15,
            dataset='../mnist.pkl.gz',
            batch_size=20, output_folder='DAE_plots'):

    print '... loading data'    
    datasets = load_data(dataset)
    
    train_set_x, train_set_y = datasets[0]

    # set number of minibatches
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    
    # Create output directory to store weight images
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    
    print '... building symbolic model'

    ############################ CREATE SYMBOLIC VARIABLES
    ##################################################################
    ##################################################################

    # create symbolic variable for the indexing minibatches
    index = T.lscalar()    # index to a [mini]batch

    # create symbolic variables for input (x represent a minibatch)
    x = T.matrix('x')  # data images


    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = DAE(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28 * 28, n_hidden=500)
    cost, updates = da.get_cost_updates(corruption_level=0., learning_rate=learning_rate)

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    #####################      TRAIN MODEL

    start_time = time.clock()

    for epoch in xrange(training_epochs):       # go through training epochs
        c = []
        for batch_index in xrange(n_train_batches):     # go through trainng set
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()
    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((training_time) / 60.))
    
    image = Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')

    #####################################
    # BUILDING THE MODEL CORRUPTION 30% #
    #####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = DAE(numpy_rng=rng, theano_rng=theano_rng, input=x, n_visible=28 * 28, n_hidden=500)
    cost, updates = da.get_cost_updates(corruption_level=0.3, learning_rate=learning_rate)

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = time.clock()

    #####################      TRAIN MODEL
    
    for epoch in xrange(training_epochs):   # go through training epochs
        c = []
        for batch_index in xrange(n_train_batches):    # go through trainng set
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()
    training_time = (end_time - start_time)

    print >> sys.stderr, ('The 30% corruption code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % (training_time / 60.))

    image = Image.fromarray(tile_raster_images(X=da.W.get_value(borrow=True).T, img_shape=(28, 28), tile_shape=(10, 10), tile_spacing=(1, 1)))
    image.save('filters_corruption_30.png')

    os.chdir('../')


if __name__ == '__main__':
    test_DAE()
