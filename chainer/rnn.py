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



"""


