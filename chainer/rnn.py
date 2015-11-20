"""
In this file I post snippets from recurrent net example on chainer's website and add notes to them
http://docs.chainer.org/en/stable/tutorial/recurrentnet.html

Recurrent Network:
- Given an input sequence x_1,x_2,...,x_t,... , with an initial state of h_0, a recurrent net iteratively updates its state as follows: h_t = f(x_t,h_{t-1}) and it outputs y_t = g(h_t)
- Expand the procedure along the time axis, the netwrok looks like a feedforward network

------
Task: 
- Given a sequence of words seen so far, predict the next word.

Input: 
- is one-hot (a vector of 1000x1 where only one item is one and the rest are zero.

There are 1,000 different words, and  we use a 100 dimensional vector to represent each word (a.k.a. word embedding).

Network:
- input is a one-hot vector telling which word it is (1000x1)
- Embedding layer: casts the one-hot notation to a 100 dimensions for each word. 
    w_{i_to_embd}: 100x1000 (each one-hot word, activates one row) 
        w_{i_to_embd} * Input + bias   ->    100x1
- Hidden layer: we take size 50 units for it.
    w_{embd_to_h}: 50x100 when multiplied by the embedding layer output generates 50 outputs.
        w_{embd_to_h} * 100x1 + bias   ->    50x1          this is h_{t-1}
- Hidden layer update: hidden to hidden (hidden to itself) weight
    w_{hidden_to_hidden}: 50x50 we want to keep multiplying it to itself many times over time, so size should be fixed.
        w_{hidden_to_hidden} * 50 + bias   ->    50x1      this is h_{t}, where we said we have h_t=f(x_t,h_{t-1})
- Output is a softmax over 1000 possibilities 
    w_{hidden_to_output}: 1000x50
        w_{hidden_to_output} * 50 + bias   -> 1000x1

===============
More on chainer functions:
    class Linear(in_size, out_size, wscale=1, bias=0, nobias=False, initialW=None, initial_bias=None)
        The weight matrix W has shape (out_size, in_size). 
        - Linear.forward(x) returns Y=XW^T+b
        - Lienar.backward(x,gy)  where gy is gradient of y, it calculates gradientes, etc
     	
	Parameters:	
	    in_size ("int"): Dimension of input "vector".
	    out_size (int): Dimension of output vector.
	    wscale (float): Scaling factor of the weight matrix.
	    bias (float): Initial bias value.
	    nobias (bool): If True, then this function does not use the bias.
	    initialW (2-D array): Initial weight value. If None, then this function uses to initialize wscale.
	    initial_bias (1-D array): Initial bias value. If None, then this function uses to initialize bias.

        It provides a funciton linear(x, W, b) to be used without class.
    

    
    class lstm(c_prev, x)
	Long Short-Term Memory units as an activation function.	This function implements LSTM units with forget gates. Let the previous cell state "cprev" and the incoming signal "x".

	First, the incoming signal x is split into four arrays a,i,f,o of the same shapes along the second axis. It means that x's second axis must have 4 times the length of cprev, where:
		a : sources of cell input (Ct_prime)
		i : sources of input gate
		f : sources of forget gate
		o : sources of output gate

	Second, it computes outputs as:
		c=tanh(a)sigmoid(i)+cprev*sigmoid(f),           # f=Wf*xt+Ufh_{t-1}+bc , so it assuems this is given. OR, others as well: 
		h=tanh(c)sigmoid(o).

	These are returned as a tuple of two variables.
	Parameters:	

	    c_prev (Variable) – Variable that holds the previous cell state. The cell state should be a zero array or the output of the previous call of LSTM.
	    x (Variable) – Variable that holds the incoming signal. It must have the second dimension four times of that of the cell state,

	Returns:	

	Two Variable objects c and h. c is the updated cell state. h indicates the outgoing signal.
	Return type:	

	tuple

	See the original paper proposing LSTM with forget gates: Long Short-Term Memory in Recurrent Neural Networks.

	Example

	Assuming y is the current input signal, c is the previous cell state, and h is the previous output signal from an lstm function. Each of y, c and h has n_units channels. Most typical preparation of x is:

	>>> model = FunctionSet(w=F.Linear(n_units, 4 * n_units),
	...                     v=F.Linear(n_units, 4 * n_units),
	...                     ...)
	>>> x = model.w(y) + model.v(h)
	>>> c, h = F.lstm(c, x)

	It corresponds to calculate the input sources a,i,f,o from the current input y and the previous output h. Different parameters are used for different kind of input sources.
     
"""

model = FunctionSet(
    embed  = F.EmbedID(1000, 100),
    x_to_h = F.Linear(100,   50),
    h_to_h = F.Linear( 50,   50),
    h_to_y = F.Linear( 50, 1000),
)
optimizer = optimizers.SGD()
optimizer.setup(model)

# The forward computation is simply as a for loop over input sequence which is a list of integer array.:

volatile=False
h = Variable(np.zeros((1, 50), dtype=np.float32), volatile=volatile)
loss   = 0
count  = 0
seqlen = len(x_list[1:])

for cur_word, next_word in zip(x_list, x_list[1:]):
    word = Variable(cur_word, volatile=volatile) # put current word into a Chainer variable
    t    = Variable(next_word, volatile=volatile) # target word (next word in sequence)
    
    x    = F.tanh(model.embed(word))
    h    = F.tanh(model.x_to_h(x) + model.h_to_h(h))
    y    = model.h_to_y(h)
    loss = F.softmax_cross_entropy(y, t)

    loss += new_loss
    count += 1
    if count % 30 == 0 or count == seqlen:   # At each 30 steps, backprop takes place at the accumulated loss.
        optimizer.zero_grads() # Do not forget to call Optimizer.zero_grads() before the backward computation!
        loss.backward()
        loss.unchain_backward() # deletes the computation history backward from the accumulated loss.
        optimizer.update()

# Note that the first dimension of h and x_list is always the mini-batch size. The mini-batch size is assumed to be 1 here. 



