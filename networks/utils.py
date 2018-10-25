import tensorflow as tf
import numpy as np

def log_normpdf(x,mu,sig):
    """
    Compute logN(x;mu,sig)
    mu, sig : (...,D)
    return  : (...)
    """
    r = (x-mu)**2/sig**2 # (...,D)
    return tf.reduce_sum(-0.5*r - 0.5*tf.log(2*np.pi) - tf.log(sig),axis=-1)

def logdet(L):
    """
    Compute logdet(X), where X = L@L^T (L: cholesky)
    L : (...,M,M)
    """
    Ldiag = tf.matrix_diag_part(L) # (...,M) 
    logLdiag = tf.log(tf.abs(Ldiag)) # (...,M)
    ldet = tf.reduce_sum(logLdiag,axis=-1) # (...,)
    return 2*ldet

def catSample(c):
    """
    Sample K categorical values from c
    """
    u = tf.random_uniform(tf.shape(c))
    g = -tf.log(-tf.log(u+1e-10))
    tau = .5
    temp = tf.exp((tf.log(c+1e-10) + g)/tau)
    return temp / tf.reduce_sum(temp,axis=-1,keepdims=True)

# def catSamples(c,B,K,T,Nc):
#     """
#     Sample K categorical values from c
#     c : (B,1,T,Nc)
#     """
#     u = tf.random_uniform((B,K,T,Nc))
#     g = -tf.log(-tf.log(u+1e-10))
#     tau = .5
#     temp = tf.exp((tf.log(c+1e-10) + g)/tau)
#     return temp / tf.reduce_sum(temp,axis=-1,keepdims=True) # (B,K,T,Nc)


def expand_tile(X,multiply):
    """
    Expand the first dimension and tile.
    X must be 2-dim tensor: e.g.(B,D).
    Return tensor has a shape of (multiply,B,D)
    """
    return tf.tile(tf.expand_dims(X,axis=0),(multiply,1,1))

def vec_to_tri(vector, N):
    indices = list(zip(*np.tril_indices(N,-1)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)
    return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

def vecs_to_tri(vectors, N):
    indices = list(zip(*np.tril_indices(N,-1)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)