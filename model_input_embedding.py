# Standard Initialization Stuff for PyTorch, Matplotlib, ...
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import numpy as np
from sklearn import preprocessing 
import torch
import torch.nn as nn
from torch.nn.functional import gumbel_softmax

# Class to load text data from a file and generate random training batches
class TextData(object):
    
    def __init__(self, name=None, path=None):
        self.all_letters = string.printable
        self.name = name
        self.data = self.load(path)
        self.reading_positions = []
        self.resets = 0
        
    # Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
    def unicodeToAscii(self, s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )

    def load(self, path):
        data_file = open(path)
        #data=list(self.unicodeToAscii(data_file.read()))
        data=list(data_file.read())
        
        data_file.close()

        self.le = preprocessing.LabelEncoder()
        self.le.fit(data)
    
        print("Extracted characters:")
        print(self.le.classes_)

        data_list = self.le.transform(data).reshape(-1,1)
        
        self.data_list = data_list
    
        self.ohe = preprocessing.OneHotEncoder()
        self.ohe.fit(data_list)
    
        data_vec = self.ohe.transform(data_list).toarray()  
        
        self.n_chars = data_vec.shape[1]
        self.length = data_vec.shape[0]
        
        print("Text Data Shape:")   
        print(data_vec.shape)
        print("Type:")
        print(type(data_vec))
        
        return data_vec;

    def slices(self, str_len, n_proc):
        # 40 50 200
        
        x = np.zeros( (str_len, n_proc, self.n_chars) )
        x_l = np.zeros( (str_len, n_proc) )
        
        #print('x_l.shape:')
        #print(x_l.shape)
        
        resets = []
        
        if len(self.reading_positions) < n_proc:
            self.reading_positions = []
            for i in range( n_proc ):
                j = np.random.randint( self.data.shape[0] - str_len )
                self.reading_positions.append(j)
                    
                x[:,i,:] = self.data[j:(j+str_len),:]
                x_l[:,i] = self.data_list[j:(j+str_len)].squeeze()
                
                resets.append(i)
                
        else:
            for i in range( n_proc ):                
                    
                j = self.reading_positions[i] + str_len

                if j + str_len > self.length:

                    self.resets = self.resets + 1

                    j = np.random.randint( self.data.shape[0] - str_len )

                    resets.append(i)

                self.reading_positions[i] = j

                x[:,i,:] = self.data[j:(j+str_len),:]
                x_l[:,i] = self.data_list[j:(j+str_len)].squeeze()
            
        return torch.FloatTensor(x).cuda(), torch.LongTensor(x_l).cuda(), resets
        
    def slices_exact(self):
        # 40 50 200
        
        x = numpy.zeros( (n_chars, n_c, n_proc_exact) )

        for i in range( n_proc_exact ):
            
            if i % n_proc_per_string_exact == 0:
                j = numpy.random.randint( self.data.shape[0] - n_chars )

            x[:,:,i] = self.data[j:(j+n_chars),:]
            
        return x
        
    def num_examples(self):
        return self.data.shape[0]
    
    def print_str(self, vec):
        print(vec.shape)
        text = np.argmax(vec.squeeze().cpu().numpy(),1)
        print(text.shape)
        translated = ''.join(self.le.inverse_transform(text).squeeze())
        print(translated)

def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std).cuda()
        m.bias.data.zero_().cuda()
        
# Class to hold the current state of a neural network, i.e. the current activation of each layer
class network_state(object):
    def __init__(self, hidden_size, n_proc):
           
        self.n_s = hidden_size
        self.n_proc = n_proc
        
        self.rnn1 = torch.zeros(self.n_proc, self.n_s).cuda()
        self.rnn2 = torch.zeros(self.n_proc, self.n_s).cuda()
        self.rnn3 = torch.zeros(self.n_proc, self.n_s).cuda()
        self.rnn4 = torch.zeros(self.n_proc, self.n_s).cuda()
        
        self.b1 = torch.ones(self.n_proc, 2).cuda()
        self.b2 = torch.ones(self.n_proc, 2).cuda()
        self.b3 = torch.ones(self.n_proc, 2).cuda()
        
        return
    
    def re_init(self):
        
        self.rnn1 = torch.zeros(self.n_proc, self.n_s).cuda()
        self.rnn2 = torch.zeros(self.n_proc, self.n_s).cuda()
        self.rnn3 = torch.zeros(self.n_proc, self.n_s).cuda()
        self.rnn4 = torch.zeros(self.n_proc, self.n_s).cuda()
        
        self.b1 = torch.ones(self.n_proc, 2).cuda()
        self.b2 = torch.ones(self.n_proc, 2).cuda()
        self.b3 = torch.ones(self.n_proc, 2).cuda()
        
        return
    
    def reset_hidden(self, resets):
        
        self.rnn1[resets,:] = 0.0
        self.rnn2[resets,:] = 0.0
        self.rnn3[resets,:] = 0.0
        self.rnn4[resets,:] = 0.0
        
        self.b1[resets,:] = 1.0
        self.b2[resets,:] = 1.0
        self.b3[resets,:] = 1.0
        
        return
    
    def detach(self):
    
        self.rnn1 = self.rnn1.detach()
        self.rnn2 = self.rnn2.detach()
        self.rnn3 = self.rnn3.detach()
        self.rnn4 = self.rnn4.detach()
        
        self.b1 = self.b1.detach()
        self.b2 = self.b2.detach()
        self.b3 = self.b3.detach()

# Class to:
#              - store the current connectivity, i.e. weights, of the network
#              - propagate the network state, given a new character input and the previous state
#              - create a prediction of the next character, given the current network state
#              - calculate the mean likelihood of a training batch
class RNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_proc, gumbel_hard):
        super(RNN, self).__init__()
        
        # Hyperparameters
        
        # Dimensionality of the state space of each layer (i.e. number of neurons at each layer)
        self.n_s = hidden_size
        # Number of training examples in a batch, to be processed in parallel
        self.n_proc = n_proc
        # Number of characters
        self.n_c = output_size
        # Dimensionality of input embedding
        self.n_e = self.n_s
        
        
        # The Bernoulli distribution of the gating units is approximated using 
        # a smooth, differentiable function, namely the gumbel-softmax. 
        # If gumbel_hard is true, the forward pass uses samples
        # from the actual Bernoulli distribution, and only the backward pass uses
        # this smooth approximation
        self.gumbel_hard = gumbel_hard

        # Weights
        
        # Hidden states at all layers to prediction of next character
        self.hg_rnns = nn.Linear(4*self.n_s, self.n_s).cuda()
        self.g_hg = nn.Linear(self.n_s,4).cuda()
        
        self.hctpi_rnns = nn.Linear(4*self.n_s, self.n_s).cuda()
        self.ctpi_hctpi = nn.Linear(self.n_s, self.n_c).cuda()
        
        # Input embedding
        self.e_ct = nn.Linear(self.n_c, self.n_e).cuda()
        
        # Update for fastest layer
        self.rnn1_b0 = nn.Linear(self.n_s + self.n_e, self.n_s ).cuda()
        self.rnn1_b1 = nn.Linear(self.n_s + self.n_e, self.n_s ).cuda()
        
        self.b1_rnn1 = nn.Linear(self.n_s, 2).cuda()
        
        # Update for second layer
        self.rnn2_b0 = nn.Linear(2*self.n_s, self.n_s).cuda()
        self.rnn2_b1 = nn.Linear(2*self.n_s, self.n_s).cuda()
        
        self.b2_rnn2 = nn.Linear(self.n_s, 2).cuda()
        
        # Update for third layer
        self.rnn3_b0 = nn.Linear(2*self.n_s, self.n_s).cuda()
        self.rnn3_b1 = nn.Linear(2*self.n_s, self.n_s).cuda()
        
        self.b3_rnn3 = nn.Linear(self.n_s, 2).cuda()
        
        # Update for slowest layer
        self.rnn4 = nn.Linear(2*self.n_s, self.n_s).cuda()
        
        # Used nonlinearities
        self.softmax = nn.LogSoftmax(dim=1).cuda()
        self.tanh = nn.Tanh().cuda()
        self.sigmoid = nn.Sigmoid().cuda()
        self.relu = nn.ReLU().cuda()

    def forward(self, ct, state, theta):
        
        # For "shortness" of notation (note: lines will still not be very short)
        
        # State of the gating units at the last timestep, i.e. character
        b1tm1 = state.b1
        b2tm1 = state.b2
        b3tm1 = state.b3
        
        # States of the individual layers at the last timestep, i.e. character
        rnn1tm1 = state.rnn1
        rnn2tm1 = state.rnn2
        rnn3tm1 = state.rnn3
        rnn4tm1 = state.rnn4
        
        # Reformated to be broadcasted correctly
        b1tm1f = b1tm1[:,0].unsqueeze(1)
        
        # Embed input character
        ct_e = self.relu( self.e_ct(ct) )
        
        # Depending on the gating unit b1 at the last timestep (active (1) or not (0)),
        # the fastest layer gets either the state of the second layer, 
        # or its own state at the last timestep as input
        rnn1b1 = self.tanh( self.rnn1_b1(torch.cat( (rnn2tm1,ct_e), dim = 1) ) )
        rnn1b0 = self.tanh( self.rnn1_b0(torch.cat( (rnn1tm1,ct_e), dim = 1) ) )
        rnn1 = b1tm1f*rnn1b1 + (1.0 - b1tm1f)*rnn1b0 
    
        # gumbel_softmax takes LOGITS (i.e. non-normalized logarithms of probabilities)
        b1t = gumbel_softmax( self.b1_rnn1( rnn1 ) , tau = theta, hard = self.gumbel_hard )
        
        # Again reformat to be broadcasted correctly
        b1tf = b1t[:,0].unsqueeze(1)
        b2tm1f = b2tm1[:,0].unsqueeze(1)
        
        # Depending on the gating unit b2 at the last timestep (active (1) or not (0)),
        # the layer gets either the state of the slower layer, 
        # or its own state at the last timestep as input
        rnn2b1 = self.tanh( self.rnn2_b1(torch.cat((rnn3tm1,rnn1), dim = 1) ) )
        rnn2b0 = self.tanh( self.rnn2_b0(torch.cat((rnn2tm1,rnn1), dim = 1) ) )
        rnn2_inner = (1.0 - b2tm1f)*rnn2b0 + b2tm1f*rnn2b1
        
        # Only update the second layer, if the gating unit at the lower layer b1 is active
        rnn2 = ( 1.0 - b1tf )*rnn2tm1 + b1tf*rnn2_inner
        
        # gumbel_softmax takes LOGITS (i.e. non-normalized logarithms of probabilities)
        b2tnew = gumbel_softmax( self.b2_rnn2( rnn2 ), tau = theta, hard = self.gumbel_hard )
        
        # Only update the gating unit b2 if the gating unit at the lower layer b1 is active
        b2t = b1tf*b2tnew + (1.0 - b1tf)*b2tm1
        
        # Again reformat to be broadcasted correctly
        b2tf = b2t[:,0].unsqueeze(1)
        b3tm1f = b3tm1[:,0].unsqueeze(1)
        
        # Depending on the gating unit b3 at the last timestep (active (1) or not (0)),
        # the layer gets either the state of the slower layer, 
        # or its own state at the last timestep as input
        rnn3b1 = self.tanh(self.rnn3_b1(torch.cat((rnn4tm1,rnn2), dim = 1) ) )
        rnn3b0 = self.tanh(self.rnn3_b0(torch.cat((rnn3tm1,rnn2), dim = 1) ) )
        rnn3_inner = b3tm1f*rnn3b1 + (1.0 - b3tm1f)*rnn3b0
        
        # Only update the third layer, if the gating units at the lower layers b1 and b2 are active
        rnn3 = (1.0 - b1tf)*rnn3tm1 + b1tf*( (1.0 - b2tf)* rnn3tm1 + b2tf*rnn3_inner )
        
        # gumbel_softmax takes LOGITS (i.e. non-normalized logarithms of probabilities)
        b3tnew = gumbel_softmax( self.b3_rnn3( rnn3 ), tau = theta, hard = self.gumbel_hard ) 
        
        # Only update the gating unit b3 if the gating units at the lower layers b1 and b2 are active
        b3t = b1tf*( b2tf*b3tnew + (1.0 - b2tf)*b3tm1 ) + (1.0 - b1tf)*b3tm1
        
        # Again reformat to be broadcasted correctly
        b3tf = b3t[:,0].unsqueeze(1)
        
        # Only update the highest layer, if all gating units, b1, b2, and b3 are active
        rnn4_inner = self.tanh( self.rnn4(torch.cat((rnn4tm1,rnn3), dim = 1)))       
        rnn4 = (1.0 - b1tf)*rnn4tm1 + b1tf*( (1.0 - b2tf)*rnn4tm1 + b2tf*( (1.0 - b3tf)*rnn4tm1 + b3tf*rnn4_inner ) )

        # Create prediction for next character using the activation at each layer
        hg = self.relu( self.hg_rnns( torch.cat( (rnn1, rnn2, rnn3, rnn4), dim = 1 ) ) )
        g = self.sigmoid( self.g_hg( hg ) )
        g1 = g[:,0].unsqueeze(1)
        g2 = g[:,1].unsqueeze(1)
        g3 = g[:,2].unsqueeze(1)
        g4 = g[:,3].unsqueeze(1)
        
        hctpi = self.relu( self.hctpi_rnns( torch.cat( (g1*rnn1, g2*rnn2, g3*rnn3, g4*rnn4), dim = 1 ) ) )
        ctpi = self.softmax( self.ctpi_hctpi(hctpi) )
        
        # Update hidden variables
        state.b1 = b1t
        state.b2 = b2t
        state.b3 = b3t
        
        state.rnn1 = rnn1
        state.rnn2 = rnn2
        state.rnn3 = rnn3
        state.rnn4 = rnn4
        
        return ctpi, state

    def first_step(self, state):
        
        # Make an initial prediction from the state of the network
        
        rnn1 = state.rnn1
        rnn2 = state.rnn2
        rnn3 = state.rnn3
        rnn4 = state.rnn4
        
        hg = self.relu( self.hg_rnns( torch.cat( (rnn1, rnn2, rnn3, rnn4), dim = 1 ) ) )
        g = self.sigmoid( self.g_hg( hg ) )
        g1 = g[:,0].unsqueeze(1)
        g2 = g[:,1].unsqueeze(1)
        g3 = g[:,2].unsqueeze(1)
        g4 = g[:,3].unsqueeze(1)
        
        hctpi = self.relu( self.hctpi_rnns( torch.cat( (g1*rnn1, g2*rnn2, g3*rnn3, g4*rnn4), dim = 1 ) ) )
        ctpi = self.softmax( self.ctpi_hctpi(hctpi) )
        
        return ctpi
    
    # weight_init
    def weight_init(self, mean, std):
        
        # Initialize the network weights with a Gaussian distribution
        
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
            
