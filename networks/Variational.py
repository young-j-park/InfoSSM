import tensorflow as tf
import numpy as np

class Variational(object):
    """
    Return p(x0|Y) ~ q(x0) = N(muhat,Sighat) and p(c|Y) ~ q(c)
    """
    def __init__(self, Dx, Dy, Nc, Dh, T):
        self.offset_Sighat = 1e-3
        self.T = T
        self.Dx = Dx
        self.Dy = Dy
        self.Nc = Nc
        self.Dh = Dh
        
        self.cell = tf.contrib.rnn.BasicRNNCell(num_units=Dh, activation=tf.nn.tanh) # set the dimension of hidden layer as twice of the state
        self.muhat_layer = tf.layers.Dense(units=Dx)
        self.logSighat_layer = tf.layers.Dense(units=Dx)
        
        self.c_layers = [tf.layers.Dense(units=64, activation=tf.nn.relu), \
                         tf.layers.Dense(units=16, activation=tf.nn.relu), \
                         tf.layers.Dense(units=Nc, activation=tf.nn.softmax)]

    def q0(self, yseq):
        B = tf.shape(yseq)[0]
        yseq_reverse = tf.reverse(yseq[:,0,:,:] ,axis=[1]) # reverse along time
        
        inputs = tf.unstack(yseq_reverse, axis=1) # transform as list
        hidden_list, hidden_initial = tf.nn.static_rnn(self.cell, inputs, dtype=tf.float32)

        muhat0 = tf.reshape(self.muhat_layer(hidden_initial), (-1,1,1,self.Dx)) # (B,1,1,Dx)
        logSighat = self.logSighat_layer(hidden_initial) # (B,Dx)
        sighat0 = tf.reshape(tf.exp(logSighat)+self.offset_Sighat,(-1,1,1,self.Dx)) # (B,1,1,Dx)
        
        # Rotation & Shift Invariant
        th = tf.random_uniform([B,1,1,1], minval=0, maxval=2*np.pi) 
        cth = tf.cos(th) # (B,1,1,1)
        sth = tf.sin(th) # (B,1,1,1)
        
        # assert (self.Dy == 2)
        pxseq_rotated = cth * yseq[:,:,:,0:1] + sth * yseq[:,:,:,1:2] # (B,1,T,1)
        pyseq_rotated = -sth * yseq[:,:,:,0:1] + cth * yseq[:,:,:,1:2] # (B,1,T,1)
        traj_seq_rotated = tf.reshape(tf.concat([pxseq_rotated, pyseq_rotated], axis=-1),(-1,self.T*self.Dy)) # (B,T*Dy)
        
        out = traj_seq_rotated
        for layer in self.c_layers:
            out = layer(out)
            
        c0 = tf.reshape(out,(-1,1,1,self.Nc)) # (B,1,1,Nc)
        
        return muhat0, sighat0, c0
                