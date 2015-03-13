"""
 Stacked Denoising Auto-Encoders (SDAE): DAEs are the building blocks for SDAE.
 Auto-Encoder: 
    Takes an input x and first maps it to a hidden representation y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. 
    The resulting latent representation y is then mapped back to a "reconstructed" vector z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b'). 
    The weight matrix W' can optionally be constrained such that W' = W^T, in which case the autoencoder is said to have tied weights. 
    The network is trained such that to minimize the reconstruction error (the error between x and z).

 DAE:
    During training, first x is corrupted into \tilde{x}, where \tilde{x} is a partially destroyed version of x by means of a stochastic mapping. 
    Afterwards y is computed as before (using \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction error is now measured between 
    z and the uncorrupted input x, which is computed as the cross-entropy :
        - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]

 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: 
   Extracting and Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103, 2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: 
   Greedy Layer-Wise Training of Deep Networks, Advances in Neural Information Processing Systems 19, 2007
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from multi_layer_perceptron import HiddenLayer
from DAE import DAE


class SDAE(object):
    """ A stacked denoising autoencoder model is obtained by stacking several DAEs. The hidden layer of the DAE at layer `i` becomes the input of the DAE at layer `i+1`. 
                The first layer DAE gets as input the input of the SDAE, and the hidden layer of the last DAE represents the output.
        After pretraining, the SDAE is dealt with as a normal MLP, the DAEs are only used to initialize the weights. This class is made to support a variable number of layers.
    """

    def __init__(self, numpy_rng, theano_rng=None,    n_ins=784, hidden_layers_sizes=[500, 500], n_outs=10,    corruption_levels=[0.1, 0.1]):
        """ numpy_rng: numpy.random.RandomState: numpy random number generator used to draw initial weights
            theano_rng: theano.tensor.shared_randomstreams.RandomStreams: Theano random generator; if None is given one is generated based on a seed drawn from `rng`
            n_ins: int: dimension of the input to the sDAE
            n_layers_sizes: list of ints: intermediate layers size, must contain at least one value
            n_outs: int: dimension of the output of the network
            corruption_levels: list of float: amount of corruption to use for each layer
        """
        self.sigmoid_layers = []
        self.DAE_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)

        assert self.n_layers > 0
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
       
        # Symbolic variables for data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.ivector('y')  # the labels are presented as 1D vector of [int] labels

        # The SDAE is an MLP, for which all weights of intermediate layers are shared with a different denoising autoencoders. We will first construct the 
        # SDAE as a deep multilayer perceptron, and when constructing each sigmoidal layer we also construct a denoising autoencoder that shares weights with that layer
        #     - During pretraining we will train these autoencoders (which will lead to changing the weights of the MLP as well)
        #     - During finetunining we will finish training the SDAE by doing stochastic gradient descent on the MLP

        # Build an MLP and a DAE in parallel to each other  with sharing weights. layer by layer
        for i in xrange(self.n_layers):
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output_data

            # hidden layer (logistic)
            sigmoid_layer = HiddenLayer(rng=numpy_rng, 
                                        input_data=layer_input, n_in=input_size, n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)    # add the layer to our list of layers
            self.params.extend(sigmoid_layer.params)
            # We are going to only declare that the parameters of the sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the DAE are parameters of those DAE, but not the SDAE

            # DAE layer with shared weights with MLP layer
            DAE_layer = DAE(numpy_rng=numpy_rng, theano_rng=theano_rng,
                          input=layer_input, n_visible=input_size, n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W, bhid=sigmoid_layer.b)  # shared weights and biases
            self.DAE_layers.append(DAE_layer)

        # Finally, Add a logistic layer at the end of the MLP
        self.logisticLayer = LogisticRegression(input_data=self.sigmoid_layers[-1].output_data, n_in=hidden_layers_sizes[-1], n_out=n_outs)
        self.params.extend(self.logisticLayer.params)

        self.finetune_cost = self.logisticLayer.negative_log_likelihood(self.y)
        self.errors = self.logisticLayer.errors(self.y)




    ####################################     PRETRAINING
    ##################################################################
    ##################################################################
    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one step in trainnig the DAE corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train a DAE you just need to iterate, calling the corresponding function on all minibatch indexes.

        train_set_x: theano.tensor.TensorType: Shared variable that contains all datapoints used for training the DAE
        batch_size: int: size of a [mini]batch
        learning_rate: float: learning rate used during training for any of the DAE layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')
        corruption_level = T.scalar('corruption')
        learning_rate = T.scalar('lr')
        
        batch_begin = index * batch_size # begining of a batch, given `index`
        batch_end = batch_begin + batch_size # ending of a batch given `index`

        pretrain_fns = []
        for DAE in self.DAE_layers:  # FOR EACH LAYER BASED ON THE LAYER LINKS DEFINED ABOVE, DETERMINE THE COST AND UPDATE FUNCTION. THS IS ACTUALLY MATERIALIZING THE FUNCTION and GLUING INPUT/OUTPUT TOGETHER
            # get the cost and the updates list
            cost, updates = DAE.get_cost_updates(corruption_level, learning_rate)
            # MATERIALIZE the theano function 
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns





    ####################################     FINE TUNING
    ##################################################################
    ##################################################################
    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train`    that implements one step of finetuning
                     a function `validate` that computes the error on a batch from the validation set
                     a function `test`     that computes the error on a batch from the testing set

        datasets: list of pairs of theano.tensor.TensorType: A list that contain all the datasets; `train`, `valid`, `test` in this order, each in tuple form (data point, label)
        batch_size: int: size of a minibatch
        learning_rate: float: learning rate used during fine tune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [(param, param - gparam * learning_rate) for param, gparam in zip(self.params, gparams)]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='train'
        )

        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: test_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='test'
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                self.y: valid_set_y[index * batch_size: (index + 1) * batch_size]
            },
            name='valid'
        )

        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in xrange(n_test_batches)]

        return train_fn, valid_score, test_score












####################################     TEST SDAE
##################################################################
##################################################################
def test_SDAE(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='../mnist.pkl.gz', batch_size=1):
    """ Demonstrates how to train and test a stochastic denoising autoencoder.

    learning_rate: float: learning rate used in the finetune stage (factor for the stochastic gradient)
    pretraining_epochs: int: number of epoch to do pretraining
    pretrain_lr: float: learning rate to be used during pre-training
    n_iter: int: maximal number of iterations to run the optimizer
    dataset: string: path the the pickled dataset
    """
    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # number of minibatches for training set
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    numpy_rng = numpy.random.RandomState(89677)
    
    ############################ CREATE SYMBOLIC VARIABLES
    ##################################################################
    ##################################################################

    print '... building the model'
    sdae = SDAE(            # construct the stacked denoising autoencoder class
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        hidden_layers_sizes=[1000, 1000, 1000],
        n_outs=10
    )
  
  
    ##############################       PRETRAINING THE MODEL
    ##################################################################
    ##################################################################
    print '... getting the pretraining functions'
    pretraining_fns = sdae.pretraining_functions(train_set_x=train_set_x, batch_size=batch_size)

    print '... pre-training the model'
    start_time = time.clock()
    corruption_levels = [.1, .2, .3]
    
    # Pre-train *****LAYER-WISE******    (THE FIRST HALF OF THE NETWORK - THE OTHER HALF WILL USE THE SAME WEIGHTS TRANSPOSED)
    
    for i in xrange(sdae.n_layers):     # go through layers (pre-train layer-after-layer)
        for epoch in xrange(pretraining_epochs):   # go through pre-training epochs
            layer_cost_in_epoch_arr = []   # to know avg cost over all minibatches
            for batch_index in xrange(n_train_batches):   # go through the training set
                layer_cost_in_epoch_arr.append(  pretraining_fns[i](index = batch_index, corruption = corruption_levels[i], lr = pretrain_lr)  )
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(layer_cost_in_epoch_arr)

    end_time = time.clock()
    print >> sys.stderr, ('The pretraining code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))



    ##############################       FINETUNING THE WEIGHTS: the shared weights are fine tuned in a simple MLP
    ##################################################################
    ##################################################################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, validate_model, test_model = sdae.build_finetune_functions(datasets=datasets, batch_size=batch_size, learning_rate=finetune_lr)

    print '... finetunning the model'
    
    # early-stopping parameters
    
    patience = 10 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is found
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)   # go through this many minibatche before checking the network on the validation set; in this case we check every epoch                


    best_validation_loss = numpy.inf
    test_score = 0.
    start_time = time.clock()

    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
           
            train_fn(minibatch_index)   # discard train minibatch_avg_cost
           
            iter_num = (epoch - 1) * n_train_batches + minibatch_index

            if (iter_num + 1) % validation_frequency == 0:
                
                # VALIDATE
                # compute zero-one loss on validation set
                validation_losses = validate_model()    # validation function defined above  TODO
                this_validation_loss = numpy.mean(validation_losses)
                
                print('epoch %i, minibatch %i/%i, validation error %f %%' % (epoch, minibatch_index + 1, n_train_batches, this_validation_loss * 100.))

                if this_validation_loss < best_validation_loss:
                    if (this_validation_loss < best_validation_loss * improvement_threshold):
                        patience = max(patience, iter_num * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter_num = iter_num

                    # test it on the test set
                    test_losses = test_model()
                    test_score = numpy.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of best model %f %%') % (epoch, minibatch_index + 1, n_train_batches, test_score * 100.))

            if patience <= iter_num:
                done_looping = True
                break

    print "LOOP FINISHED\n"
    end_time = time.clock()
    print('Optimization complete with best validation score of %f %%, on iteration %i, with test performance %f %%' % (best_validation_loss * 100., best_iter_num + 1, test_score * 100.))
    print >> sys.stderr, ('The training code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))


if __name__ == '__main__':
    test_SDAE()
