import tensorflow as tf
import numpy as np
import pickle
from .utils import *
from .MGP import * 
from .Observation import * 
from .Variational import *

class InfoSSM(object):
    """
    B    := N_batch
    K    := N_sample
    T    := N_step
    M    := N_inducingPoints
    Dx   := dim_x
    Dy   := dim_y
    Din  := dim_GPin
    Dout := dim_GPout
    Nc   := num_code
    dt   := step interval
    Matrix variables have a shape of (B,K,T,-,-).
    Vetors variables have a shape of (B,K,T,-,1).
    """
    def __init__(self, K, T, M, Dx, Dy, Nc, dt, lr):
        ### Initialize
        self.K = K
        self.T = T
        self.M = M
        self.Dx = Dx
        self.Dy = Dy
        self.Din = Dy
        self.Dout = Dy
        self.Nc = Nc
        self.dt = dt
        self.lr = lr
        self.code_transitoin = False
        
        self.info_lambda = tf.placeholder(tf.float32, shape=()) # (B,1,T,Dy)
        self.yseq_true = tf.placeholder(tf.float32, shape=(None,1,self.T,self.Dy)) # (B,1,T,Dy)
        self.y0_true = self.yseq_true[:,:,0:1,:]
        self.yseq = self.yseq_true - self.y0_true
        self.B = tf.shape(self.yseq_true)[0]
        
        ### Create Network
        self._createNetwork()
        
        ### Corresponding optimizer
        self._create_loss_optimizer()
        
        ### Initializing the tensor flow variables and saver
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        
        ### Launch the session
        self.sess.run(init)
        
    def _createNetwork(self):
        # 0. Build Dynamics and Sensor Network
        self.dynNets = []
        for i in range(self.Nc):
            self.dynNets.append(MGP(self.Din, self.Dout, self.M))
        self.senNet = Observation(self.Dx, self.Dy)
        self.varNet = Variational(self.Dx, self.Dy, self.Nc, 16, self.T)
    
        self.p0_mean = 0.0
        self.p0_sig  = 1.0
        
        if self.code_transitoin:
            self.TM_code_logvar = tf.Variable(tf.eye(self.Nc, batch_shape=(1,1))*3.0 - 3.0, dtype=tf.float32)  # transition matrix of the code
            self.TM_code_var = tf.exp(self.TM_code_logvar)
            self.TM_code = self.TM_code_var / tf.reduce_sum(self.TM_code_var,axis=-1, keepdims=True)
        else:
            self.TM_code = tf.eye(self.Nc, batch_shape=(1,1), dtype=tf.float32)  # transition matrix of the code
        

    def _create_loss_optimizer(self):
        self.x0_mean, self.x0_sig, self.c0 = self.varNet.q0(self.yseq)
        
        # 1. Sample x0, c0 ~ q0
        epsilon0 = tf.random_normal([self.B,self.K,1,self.Dx]) # (B,K,1,Dx)
        self.x0_samples = self.x0_mean + self.x0_sig*epsilon0 # (B,K,1,Dx)
        self.y0_samples = self.senNet.observe(self.x0_samples) # (B,K,1,Dy)
        ctile = tf.tile(self.c0, (1,self.K,1,1)) # (B,K,1,Nc)
        self.c0_samples = catSample(ctile) # (B,K,1,Nc)
        
        # 2. Compute the Initial Cost
        self.logqx0 = tf.squeeze(log_normpdf(self.x0_samples, self.x0_mean, self.x0_sig),-1) # (B,K)
        self.logpx0 = 0.0*tf.squeeze(log_normpdf(self.x0_samples, self.p0_mean, self.p0_sig),-1) # (B,K)
        
        self.logqc0 = tf.reduce_sum(tf.log(tf.reduce_sum(self.c0 * self.c0_samples, axis=-1)),axis=-1) # (B,K)
        self.logpc0 = tf.log(1/self.Nc) # (scalar)

        si0 = self.logqx0 - self.logpx0 + self.logqc0 - self.logpc0 # initial cost
        ss0 = -self.senNet.loglik(self.yseq[:,:,0:1,:], self.y0_samples) # state cost
        S = si0 + ss0
        log_weight = -S - tf.log(self.K*1.0) # (B,K)
        log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True) # (B,1)
        log_weight = log_weight-log_norm # (B,K)
        
        bound = log_norm # (B,1)
        
        # 3. Predict {x(t)} from x0, c0
        xt = self.x0_samples
        xseq = xt
        ct = self.c0_samples
        cseq = ct
        yseq = self.y0_samples
        for t in range(self.T-1):
            # Propagate
            ct, Sc = self.switch_mode(ct)
            
            xt_reshape = tf.reshape(xt, (-1,self.Dx)) # (BK,Dx)
            ct_reshape = tf.reshape(ct, (-1,self.Nc)) # (BK,Nc)
            
            xt_next = self.propagate(xt_reshape, ct_reshape) # (BK,Dx)
            xt = tf.reshape(xt_next, (-1, self.K, 1, self.Dx)) # (B,K,1,Dx)
            yt = self.senNet.observe(xt) # (B,K,1,Dy)
            
            xseq = tf.concat([xseq, xt], axis=2) # (B,K,t+2,Dx)
            cseq = tf.concat([cseq, ct], axis=2) # (B,K,t+2,Nc)
            yseq = tf.concat([yseq, yt], axis=2) # (B,K,t+2,Dy)
                                     
            St = Sc - self.senNet.loglik(self.yseq[:,:,t+1:t+2,:], yt) # (B,K)
            
            log_weight = log_weight - St # (B,K)
            log_norm = tf.reduce_logsumexp(log_weight, axis=1, keepdims=True) # (B,1)
            log_weight = log_weight - log_norm # (B,K)
            
            bound = bound + log_norm
            S = S + St # (B,K)
        
        self.xseq_samples = tf.concat([xseq[:,:,:,:self.Dy]+self.y0_true, xseq[:,:,:,self.Dy:]], axis=-1) # (B,K,T,Dx)
        self.yseq_samples = yseq + self.y0_true # (B,K,T,Dy)
        
        # 4. Compute the KL(q(U)||p(U))
        self.KL_term = tf.constant(0.0, dtype=tf.float32)
        for i in range(self.Nc):
            self.KL_term = self.KL_term + self.dynNets[i].KL()
              
        # 5. Compute the Info                      
        dist = tf.contrib.distributions.OneHotCategorical(probs=np.ones((self.Nc))/self.Nc, dtype=tf.float32)
        cnoise0 = dist.sample(self.B*self.K) # (BK,Nc)            
        
        xnoise = tf.reshape(tf.stop_gradient(self.x0_samples), (-1,self.Dx)) # (BK,Dx)
        xnoise_list = [xnoise]
        cnoise_list = [cnoise0]
        cnoise = cnoise0
        
        for t in range(self.T-1):
            if self.code_transitoin:
                c_prob = tf.squeeze(tf.reshape(cnoise, (self.B*self.K,1,self.Nc))@tf.tile(self.TM_code[0], (self.B*self.K,1,1)), axis=1) # (BK,Nc)
                cnoise = catSample(c_prob) # (BK,Nc)
            xnoise = self.propagate(xnoise, cnoise) # (BK,Dx)
            xnoise_list.append(xnoise)
            cnoise_list.append(cnoise)
            
        self.cnoise_seq = tf.stack(cnoise_list, axis=1)                         
        xnoise_seq = tf.reshape(tf.stack(xnoise_list, axis=1),(-1, self.T, self.Dx)) # (BK,T,Dx)                       
        ynoise_seq = tf.reshape(self.senNet.observe(xnoise_seq, True),(self.B*self.K,1,self.T,self.Dy)) # (B,K,T,Dy)
        _, _, c_noise_infer = self.varNet.q0(ynoise_seq)  # (B*K,1,1,Nc)         
        
        Infoprob = tf.reduce_sum(cnoise0 * c_noise_infer[:,0,0,:], axis=-1)
        self.Info = tf.reduce_sum(tf.log(Infoprob + 1e-10)) / tf.cast(self.B*self.K, dtype=tf.float32) # (scalar) pcjump[:,0,0]**(self.T-1) * 

        # 6. Compute the loss
        self.loss = -tf.reduce_sum(bound) + self.KL_term - self.info_lambda * self.Info
        
        # 7. Define opt
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_norm(grad, 1.0), var) for grad, var in grads_and_vars]
        self.opt = self.optimizer.apply_gradients(clipped_grads_and_vars)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
    
    def switch_mode(self, c):
        c_prob = c@tf.tile(self.TM_code, (self.B,self.K,1,1)) # (B,K,1,Nc)
        c_samples = catSample(c_prob) # (B,K,1,Nc)
        Sc = tf.log(tf.reduce_sum(c_samples * c_prob, axis=[-1,-2])) # (B,K)
        return c_samples, Sc 
        
    def propagate(self, x, c):
        """
        User need to define the shape of dynamics.
        
        x : (BK,Dx) tensor
        c : (BK,Nc) tensor
        """
        xdot1 = x[:,self.Dy:2*self.Dy] # (BK,Dy)
        xdot2 = tf.zeros_like(xdot1)
        for i in range(self.Nc):
            xdot2 = xdot2 + c[:,i:i+1]*self.dynNets[i].predict(x[:,self.Dy:2*self.Dy]*3)/3 # (BK,Dy)
        xdot = tf.concat([xdot1, xdot2], axis=1) # (BK,Dx)

        return x + xdot*self.dt # (BK,Dx)

    
    def train(self,yseq_true, info_lambda):
        _, loss, kl, Info = self.sess.run([self.opt, self.loss, self.KL_term, self.Info], \
                                          feed_dict={self.yseq_true:yseq_true, self.info_lambda:info_lambda})
        return loss, kl, Info
    
    def saveWeights(self, filename="weights.pkl"):
        """Save the weights of InfoSSM networks"""
        weights = {}
        for i, gp in enumerate(self.dynNets):
            weights['d'+str(i)+'_ARD_loglambda'] = self.sess.run(gp.ARD_loglambda)
            weights['d'+str(i)+'_ARD_logsig0'] = self.sess.run(gp.ARD_logsig0)
            weights['d'+str(i)+'_Z'] = self.sess.run(gp.Z)
            weights['d'+str(i)+'_U_mean'] = self.sess.run(gp.U_mean)
            weights['d'+str(i)+'_U_logL_diag'] = self.sess.run(gp.U_logL_diag)
            weights['d'+str(i)+'_U_L_nondiag'] = self.sess.run(gp.U_L_nondiag)
            if gp.D_OUT == 1:
                weights['d'+str(i)+'_Sig_logL'] = self.sess.run(gp.Sig_logL)
            else:
                weights['d'+str(i)+'_Sig_logL_diag'] = self.sess.run(gp.Sig_logL_diag)
                weights['d'+str(i)+'_Sig_L_nondiag'] = self.sess.run(gp.Sig_L_nondiag)
            weights['d'+str(i)+'_logbeta'] = self.sess.run(gp.logbeta)
        
        weights['s_logsig'] = self.sess.run(self.senNet.logsig)
        
        weights['v_rnn'] = self.sess.run(self.varNet.cell.weights)
        for i, layer in enumerate(self.varNet.c_layers):
                weights['v_c'+str(i)] = self.sess.run(layer.weights)    
        weights['v_mu'] = self.sess.run(self.varNet.muhat_layer.weights)
        weights['v_logSighat'] = self.sess.run(self.varNet.logSighat_layer.weights)
  
        with open(filename, 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('weight saved in '+filename)
    
    def restoreWeights(self, filename="weights.pkl"):
        """Load the weights of InfoSSM networks"""
        
        with open(filename, 'rb') as handle:
            weights = pickle.load(handle)
        
        for j, gp in enumerate(self.dynNets):
            i = j #0
            self.sess.run(tf.assign(gp.ARD_loglambda, weights['d'+str(i)+'_ARD_loglambda']))
            self.sess.run(tf.assign(gp.ARD_logsig0, weights['d'+str(i)+'_ARD_logsig0']))
            self.sess.run(tf.assign(gp.Z, weights['d'+str(i)+'_Z']))
            self.sess.run(tf.assign(gp.U_mean, weights['d'+str(i)+'_U_mean']))
            self.sess.run(tf.assign(gp.U_logL_diag, weights['d'+str(i)+'_U_logL_diag']))
            self.sess.run(tf.assign(gp.U_L_nondiag, weights['d'+str(i)+'_U_L_nondiag']))
            if gp.D_OUT == 1:
                self.sess.run(tf.assign(gp.Sig_logL, weights['d'+str(i)+'_Sig_logL']))
            else:
                self.sess.run(tf.assign(gp.Sig_logL_diag, weights['d'+str(i)+'_Sig_logL_diag']))
                self.sess.run(tf.assign(gp.Sig_L_nondiag, weights['d'+str(i)+'_Sig_L_nondiag']))
            self.sess.run(tf.assign(gp.logbeta, weights['d'+str(i)+'_logbeta']))
            
        self.sess.run(tf.assign(self.senNet.logsig, weights['s_logsig']))
        
        for j, w in enumerate(self.varNet.cell.weights):        
                self.sess.run(tf.assign(w, weights['v_rnn'][j]))
        for i, layer in enumerate(self.varNet.c_layers):
            for j, w in enumerate(layer.weights):
                self.sess.run(tf.assign(w, weights['v_c'+str(i)][j]))
        for j, w in enumerate(self.varNet.muhat_layer.weights):
                self.sess.run(tf.assign(w, weights['v_mu'][j]))
        for j, w in enumerate(self.varNet.logSighat_layer.weights):
                self.sess.run(tf.assign(w, weights['v_logSighat'][j]))
        
        print('weight restored from '+filename)