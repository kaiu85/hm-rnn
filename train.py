# Standard Initialization Stuff for PyTorch, Matplotlib, ...
from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch.optim as optim
import numpy as np
import scipy
from sklearn import preprocessing 
import torch
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax
import random
from torch.distributions.categorical import Categorical

from model_input_embedding import TextData, network_state, RNN

##### PARAMETERS

# Continue training with saved model
continue_training = False

# Training data
data_file = 'german_u.txt' #'/home/kai/bernstein/corpora/wikipedia/scripts/dewiki.txt'

save_name = 'HM_RNN_GERMAN'

# Length of unrolled strings for training
seq_len = 300
# Learning rate for ADAM optimiser
lr_start = 3e-4
lr_end = 1e-5
lr_tau = 1e-4
# Number of parallel training examples (i.e. size of the mini-batches)
n_proc = 100
# Dimensionality of each hidden layer
hidden_size = 300
#Number of optimization steps
n_steps = 5000000000

# Clip gradient?
clip_grad = True
clip_norm = 1.0

# Parameters of Gumbel-Softmax (i.e. differentiable approximation of Bernoulli-Distribution)
# True: Use actual samples from Bernoulli for the forward pass, use smooth approximation for the backward pass
# False: Use smooth approximation for the forward and the backward pass
gumbel_hard = True

# Smoothness parameter: Has to be positive, larger == smoother
theta_start = 1.0
theta_end = 0.01
theta_tau = 1e-4

# Which GPU to use
torch.cuda.set_device(0)

##### Manages the reading of a text file, the conversion between
##### characters, labels and one-hot encodings
##### and the ***sequential*** reading of sliced mini-batches

data = TextData(path=data_file)


##### Just some tests of the data-management class
# shape of the data
print(data.data.shape)

# get some example batches
example, example_l, resets = data.slices(200,10)
example2, example2_l, resets = data.slices(200,10)
example3, example3_l, resets = data.slices(200,10)

# check if the successive reading over batches works by printing the
# first sample of three successive training batches
print(example.shape)
data.print_str(example[:,0,:])
print(example2.shape)
data.print_str(example2[:,0,:])
print(example3.shape)
data.print_str(example3[:,0,:])

# initialize network state and connectivity
input_size = data.n_chars
output_size = input_size

state = network_state(hidden_size, n_proc)

rnn = RNN(hidden_size, data.n_chars, n_proc, gumbel_hard)

# initialize weights with Gaussian distribution with 
# mean zero and std dev 0.05
rnn.weight_init(0.0, 0.05)

# if a network was saved before, load it and continue training it
if continue_training == True:
    rnn.load_state_dict(torch.load(save_name + '.pkl'))

# functional form of objective function: Categorical Negative Log Likelihood
criterion = nn.NLLLoss().cuda()

# optimizer: Adam (https://arxiv.org/abs/1412.6980)
optimizer = optim.Adam(rnn.parameters(), lr=lr_start, betas=(0.5, 0.999))

# perform a training step on a training batch
def train(data, n_proc, state, theta, print_segmentation = True):
	
	# set rnn to training mode (in case some previous code called rnn.eval() )
    rnn.train()
    
    # get a  training batch
    batch, batch_l, resets = data.slices(seq_len, n_proc)
    
    # reset hidden states, if some of the reading processes encountered the end
    # of the training data
    state.reset_hidden(resets)
    
    # reset the gradients
    rnn.zero_grad()
    
    # this will be the cost function, that we take the derivative of
    cost = 0
    
    # some code to output the activation of the gating units b1, b2, b3
    if print_segmentation == True:
        outputs1 = []
        outputs2 = []
        outputs3 = []
    
    # create a prediction of the first characters from the current states of the
    # hidden layers
    output = rnn.first_step(state)
    
    # calculate the negative log-likelihood (NLL) of the first characters under this prediction
    l = criterion(output, batch_l[0])
    
    # add this NLL to the cost
    cost += l
        
    # iterate over all but the last character in the training data
    for i in range(seq_len - 1):
		
		# propagate the rnn, given the current character and state
		# and the current "temperature" parameter theta of the gumbel-softmax
		# distribution (the lower, the closer it is to a true bernoulli distribution)
		# returns a predictive probability distribution for the next character
        output, state = rnn(batch[i], state, theta)
        
        # add the nll of the next character under the predictive distribution
        l = criterion(output, batch_l[i+1])
        
        # add this to the cost
        cost += l
        
        # some output code to examine the activity of the gating units
        if print_segmentation == True:
        
            next_char = data.le.inverse_transform(np.argmax(batch[i,0].cpu().numpy())).squeeze()
        
            outputs1.append(next_char)
            outputs2.append(next_char)
            outputs3.append(next_char)
            
            if state.b1[0,0].detach().cpu().numpy() > 0.5:
                outputs1.append('|')
                
            if state.b1[0,0].detach().cpu().numpy() > 0.5 and state.b2[0,0].detach().cpu().numpy() > 0.5:
                outputs2.append('|')
                
            if state.b1[0,0].detach().cpu().numpy() > 0.5 and state.b2[0,0].detach().cpu().numpy() > 0.5 and state.b3[0,0].detach().cpu().numpy() > 0.5:
                outputs3.append('|')
    
    # Update hidden representations with the last letter in the sequence!!!!
    # This is important, since the first letter of the next training batch
    # will be predicted from these hidden states.
    output, state = rnn(batch[seq_len - 1], state, theta)
    
    # Calculate the derivative of the sum of all the NLLs w.r.t. the
    # parameters of the RNN
    cost.backward()
    
    # In RNNs, sometimes gradients can grow exponentially. So restrict the
    # norm of the gradient, just to be sure.
    if clip_grad:
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), clip_norm)
    
    # Change the parameters of the RNN, given their current values, gradients
    # and some internal variables kept by the optimizer
    optimizer.step()
    
    # Detach the current state, so that the gradients of the next training batch
    # will not be backpropagated through this state.
    state.detach()
    
    # Print output segmented by the activity of the gating units
    if print_segmentation == True:
        print('\nTRAINING DATA Segmented according to b1:\n')
        print(''.join(outputs1))
        print('\nTRAINING DATA Segmented according to b2:\n')
        print(''.join(outputs2))
        print('\nTRAINING DATA Segmented according to b3:\n')
        print(''.join(outputs3))
        
        
    # Return the numerical value of the cost function, 
    # and the current hidden state, to be used as initial state for the next batch
    return cost.detach().cpu().numpy(), state

# very similar to the training function, only that instead of evaluating the 
# negative log-likelihood of the next letter from the training data,
# the next letter is just sampled from the predictive density
# Note that the initial hidden state will be conditioned by a starting string,
# which by default is 'Hallo'. An empty string should be possible, but not
# tested yet.
def sample(start_letters = 'Hallo', theta = 0.1):
	
	# We don't need PyTorch to keep track of relations required to compute
	# gradients later on
    with torch.no_grad():
		
		# Also the rnn needs some less book-keeping if we tell it, that we
		# don't want to optimize it
        rnn.eval()
        
        # Initialize a single hidden state
        sampling_state = network_state(hidden_size, 1)
    
		
        if len(start_letters) > 0:
            start_list = data.le.transform(list(start_letters)).reshape(-1,1)
            start_tensor = torch.FloatTensor(data.ohe.transform(start_list).toarray()).cuda()
    
            for i in range(len(start_letters)):
                output, sampling_state = rnn(start_tensor[i].unsqueeze(0), sampling_state, theta)
        else:
            output = rnn.first_step(sampling_state)
        
        outputs1 = []
        outputs2 = []
        outputs3 = []
        
        # Generate seq_len samples
        for i in range(seq_len):
			
			# Create a categorical distribution with probabilities given by the predictive distribution
            dist = Categorical(logits = output)
            
            # Sample a character from that distribution
            new_input = dist.sample()
            
            # Convert the one-hot encoding to a standard character, to be able to print it
            next_char = data.le.inverse_transform(new_input.cpu().numpy()[0]).squeeze()
            
            # Append the new character to all output strings
            outputs1.append(next_char)
            outputs2.append(next_char)
            outputs3.append(next_char)
            
            # Add a segmentation mark, when the fastest gating unit b1 is active
            if sampling_state.b1.cpu().numpy()[0,0] > 0.5:
                outputs1.append('|')
                
            # Add a segmentation mark, when the gating units b1 and b2 are active
            if sampling_state.b1.cpu().numpy()[0,0] > 0.5 and sampling_state.b2.cpu().numpy()[0,0] > 0.5:
                outputs2.append('|')
               
            # Add a segmentation mark, when all gating units b1, b2, b3 are active
            if sampling_state.b1.cpu().numpy()[0,0] > 0.5 and sampling_state.b2.cpu().numpy()[0,0] > 0.5 and sampling_state.b3.cpu().numpy()[0,0] > 0.5:
                outputs3.append('|')
            
            # Send the new character also to the GPU to generate the next prediction
            x = data.ohe.transform(new_input.cpu().numpy().reshape(-1,1))
            new_input = torch.FloatTensor(x.todense()).cuda()
            
            # Propagate the RNN with the new character
            output, sampling_state = rnn(new_input, sampling_state, theta)
        
        # Print output strings
        print('\nSAMPLES Segmented according to b1:\n')
        print(''.join(outputs1))
        print('\nSAMPLES Segmented according to b2:\n')
        print(''.join(outputs2))
        print('\nSAMPLES Segmented according to b3:\n')
        print(''.join(outputs3))
    
# Initialize the temperature of the gumbel-softmax
# higher == smoother approximation of true Bernoulli distribution
theta = theta_start

# Iterate over training steps
for i in range(n_steps):
    
    # Anneal learning rate exponentially from lr_start to lr_end with
    # time constant lr_tau
    lr = (lr_start - lr_end)*np.exp(-i*lr_tau) + lr_end
    
    # Set the current learning rate
    optimizer.param_groups[0]['lr'] = lr
    
    # Anneal the current temperature of the gumbel softmay exponentially
    # from theta_start to theta_end with time constant theta_tau
    theta = (theta_start - theta_end)*np.exp(-i*theta_tau) + theta_end
    
    # Print segmented input every 100th iteration
    if i % 100 == 0:
        cost, state = train(data,n_proc,state,theta)
    else:
        cost, state = train(data,n_proc,state,theta,print_segmentation=False)
        
    # Generate a sample every 100th iteration
    if i % 100 == 0:
        sample()
        print("crit: %f" % (100.0*cost/seq_len) )
        print('theta: %f' % theta)
        
    # Save the parameters of the RNN every 1000th iteration
    if i % 1000 == 0:
        torch.save(rnn.state_dict(), save_name + '.pkl')
      
    # Save the current cost at every iteration  
    if i == 0:
        with open('./plots/log_' + save_name + '.txt', "w") as myfile:
            myfile.write("%f\n" % (100.0*cost/seq_len) )
    else:
        with open('./plots/log_' + save_name + '.txt', "a") as myfile:
            myfile.write("%f\n" % (100.0*cost/seq_len) )
