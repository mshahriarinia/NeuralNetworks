   ######################################################################################

`class Linear(in_size, out_size, wscale=1, bias=0, nobias=False, initialW=None, initial_bias=None)`

The weight matrix `W` has shape `(out_size, in_size)`.

- `Linear.forward(x)` returns `Y=XW^T+b`

- `Linear.backward(x,gy)`  where `gy` is gradient of `y`, it calculates gradientes, etc

Parameters:

```
in_size ("int"): Dimension of input "vector".
out_size (int): Dimension of output vector.
wscale (float): Scaling factor of the weight matrix.
bias (float): Initial bias value.
nobias (bool): If True, then this function does not use the bias.
initialW (2-D array): Initial weight value. If None, then this function uses to initialize wscale.
initial_bias (1-D array): Initial bias value. If None, then this function uses to initialize bias.
```
        
It provides a funciton `linear(x, W, b)` to be used without class.

   ######################################################################################

In an LSTM we have the following gates:

```latex
  i_t         = sigma(W_i* xt + U_i * h_{t-1} + b_i)
	\tilde{C}_t =  tanh(W_C* xt + U_C * h_{t-1} + b_C)        # in Chainer terms \tilde{C}_t === a
  f_t         = sigma(W_f* xt + U_f * h_{t-1} + b_f)
  o_t         = sigma(W_o* xt + U_o * h_{t-1} + b_o + V_o * C_t)

	// to have outputs: 
  C_t = i_t * \tilde{C}_t + f_t * C_{t-1}
  h_t = o_t * tanh(C_t)

```

So, in Chainer, we concatenate `[W_i, W_C, W_f, W_o]`  as `'w'` in the Example 1 below, and concatenate `[U_i, U_C, U_f, U_o]` as `'v'` in Example 1 below.
				
Then multiply 'w' to y (current input signal) and multiply `'v'` to `h` (previous "output" signal)(same as above `h_{t-1}`). O.K.
				
Then feed them to LSTM to do the sigmas and tanhs and print out `C_t` and `h_t`.
				
NOTE: Chainer's implementation is missing the `V_o * C_t` term. No big deal just be mindful of that. It caused a bit of a confusion.		 	 	  

#class LSTM
	
It has two inputs `(c, x)` and two outputs `(c, h)`, where
		
`c` indicates the cell state.
		
`x` must have four times channels compared to the number of units. (concatenation of four weight matrices used in LSTM.

```python
	LSTM.forward(c_prev, x):
		a, i, f, o = _extract_gates(x)
			# these are the weight matrices, where
		  # _extract_gates(x):  r = x.reshape( (x.shape[0], x.shape[1] // 4, 4) + x.shape[2:])     return (r[:, :, i] for i in six.moves.range(4))

			# First, the incoming signal x is split into four arrays a,i,f,o of the same shapes along the second axis. It means that x's second axis must have 4 times the length of cprev, where:
			#     `a` : sources of cell input (Ct_prime)
			#     `i` : sources of input gate
			#     `f` : sources of forget gate
			#     `o` : sources of output gate

			# Then, it computes outputs as:
	c = tanh(a) * sigmoid(i) + c_prev * sigmoid(f)           # `f=Wf*xt+Ufh_{t-1}+bc` , so it assuems this is given. OR, others as well:
	h = tanh(c) * sigmoid(o)

	return (c,h)   # `h` is the output signal, `c` is the updated cell state.
```     
####-------------------------------------------------------------------------------------
  *Example*:
  
	    Assume `y` is the current input signal, `c` is the previous cell state, and `h` is the previous output signal from an lstm function. (`y`, `c` and `h` have `n_units` channels). Most typical preparation of `x` is:

```python	 
				>>> model = FunctionSet(w=F.Linear(n_units, 4 * n_units),
				...                     v=F.Linear(n_units, 4 * n_units),
				...                     ...)
				>>> x = model.w(y) + model.v(h)    # this h is h_{t-1}
				>>> c, h = F.lstm(c, x)            # this h is h_t
```

It corresponds to calculate the input sources `a`,`i`,`f`,`o` from the current input `y` and the previous output `h`. Different parameters are used for different kind of input sources.

