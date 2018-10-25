import tensorflow as tf
import numpy as np

class Observation(object):
    """ Fixed generative model: y=g(x)=x[:,0] """
    def __init__(self, Dx, Dy):
        self.Dx = Dx
        self.Dy = Dy
        
        self.logsig = tf.Variable(np.array([-6.9078,-6.9078]), dtype=tf.float32)
        self.obs_sig = tf.exp(self.logsig)
#         self.obs_sig = tf.clip_by_value(tf.exp(self.logsig), tf.constant(1e-2, dtype=tf.float32), tf.constant(1, dtype=tf.float32))
        
    def observe(self, x, noise=False):
        # Input : x=(...,Dx)
        # Output: y=(...,Dy)
        y, _ = tf.split(x, [self.Dy, self.Dx-self.Dy], -1)
        
        if noise:
            return y + self.obs_sig * tf.random_normal(tf.shape(y))
        else:
            return y
    
    def loglik(self, ytrue, y):
        # Input : y =(...,T,Dy)
        # Output: ll=(...,)
        ll = -0.5*(ytrue-y)**2/self.obs_sig**2 - 0.5*tf.log(2*np.pi) - tf.log(self.obs_sig)
        return tf.reduce_sum(ll, axis=[-1,-2])
    
# class Observation(object):
#     """ Fixed generative model: y=g(x)=x[:,0] """
#     def __init__(self, Dx, Dy):
#         self.Dx = Dx
#         self.Dy = Dy
#         self.obs_Rsig = 1e-2 # ~10m
#         self.obs_THsig = 1e-3 # !1 deg
    

#     def observe(self, x):
#         # Input : x=(...,Dx)
#         # Output: y=(...,Dy)
#         px, py, _ = tf.split(x, [1,1, self.Dx-2], -1)
        
#         r = tf.sqrt(px**2 + py**2)
#         th = tf.atan2(py,px)
        
#         return tf.concat([r, th], axis=-1)
    
#     def loglik(self, ytrue, yest):
#         # Input : y =(...,T,Dy)
#         # Output: ll=(...,)
#         rest, thest = tf.split(yest, [1,1], -1)
#         rtrue, thtrue = tf.split(ytrue, [1,1], -1)
        
#         ll_r = -0.5*(rtrue-rest)**2/self.obs_Rsig**2 - 0.5*tf.log(2*np.pi) - tf.log(self.obs_Rsig)
        
#         dth = np.pi - tf.abs(np.pi - tf.abs(thtrue-thest))
#         ll_th = -0.5*(dth)**2/self.obs_THsig**2 - 0.5*tf.log(2*np.pi) - tf.log(self.obs_THsig)
        
#         return tf.reduce_sum(ll_r + ll_th, axis=[-1,-2])
    
#     def observe(self, x):
#         # Input : x=(...,Dx)
#         # Output: y=(...,Dy)
#         pxtrue, pytrue, _ = tf.split(x, [1,1, self.Dx-2], -1)
        
#         rtrue = tf.sqrt(pxtrue**2 + pytrue**2)
#         robs = rtrue + self.obs_Rsig * tf.random_normal(tf.shape(rtrue))
        
#         thtrue = tf.atan2(pytrue,pxtrue)
#         thobs = thtrue + self.obs_THsig * tf.random_normal(tf.shape(thtrue))
        
#         pxobs = robs * tf.cos(thobs)
#         pyobs = robs * tf.sin(thobs)
        
#         return tf.concat([pxobs, pyobs], axis=-1)
    
#     def loglik(self, ytrue, yest):
#         # Input : y =(...,T,Dy)
#         # Output: ll=(...,)
#         pxest, pyest = tf.split(yest, [1,1], -1)
#         rest = tf.sqrt(pxest**2 + pyest**2)
#         thest = tf.atan2(pyest,pxest)
        
#         rtrue, thtrue = tf.split(ytrue, [1,1], -1)
        
#         ll_r = -0.5*(rtrue-rest)**2/self.obs_Rsig**2 - 0.5*tf.log(2*np.pi) - tf.log(self.obs_Rsig)
        
#         dth = np.pi - tf.abs(np.pi - tf.abs(thtrue-thest))
#         ll_th = -0.5*(dth)**2/self.obs_THsig**2 - 0.5*tf.log(2*np.pi) - tf.log(self.obs_THsig)
        
#         return tf.reduce_sum(ll_r + ll_th, axis=[-1,-2])