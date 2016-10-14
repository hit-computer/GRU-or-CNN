#coding:utf-8
import theano, random
import numpy as np
import theano.tensor as T
from collections import OrderedDict

def sharedX(value, name=None, borrow=False, dtype=None):
    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(theano._asarray(value, dtype=dtype),
                         name=name,
                         borrow=borrow)

def Adam(grads, lr=0.0002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    i = sharedX(0.)
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in grads.items():
        m = sharedX(p.get_value() * 0.)
        v = sharedX(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates
    
def compute_updates(training_cost, params, config):
    updates = []
     
    grads = T.grad(training_cost, params)
    grads = OrderedDict(zip(params, grads))

    # Clip stuff
    c = np.float32(1.)
    clip_grads = []
    
    norm_gs = T.sqrt(sum(T.sum(g ** 2) for p, g in grads.items()))
    normalization = T.switch(T.ge(norm_gs, c), c / norm_gs, np.float32(1.))
    notfinite = T.or_(T.isnan(norm_gs), T.isinf(norm_gs))
     
    for p, g in grads.items():
        clip_grads.append((p, T.switch(notfinite, np.float32(.1) * p, g * normalization)))
    
    grads = OrderedDict(clip_grads)

    updates = Adam(grads, config.learning_rate) #使用adam梯度更新策略

    return updates
    
def NormalInit(rng, sizeX, sizeY, scale=0.01, sparsity=-1):
    """ 
    Normal Initialization
    """
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    
    if sparsity < 0:
        sparsity = sizeY
     
    sparsity = np.minimum(sizeY, sparsity)
    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
        
    return values.astype(theano.config.floatX)
    
def OrthogonalInit(rng, sizeX, sizeY, sparsity=-1, scale=1):
    """ 
    Orthogonal Initialization
    """

    sizeX = int(sizeX)
    sizeY = int(sizeY)

    assert sizeX == sizeY, 'for orthogonal init, sizeX == sizeY'

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = np.minimum(sizeY, sparsity)

    values = np.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals

    # Use SciPy:
    if sizeX*sizeY > 20000000:
        import scipy
        u,s,v = scipy.linalg.svd(values)
    else:
        u,s,v = np.linalg.svd(values)
    values = u * scale
    return values.astype(theano.config.floatX)
    
def SoftMax(x):
    x = T.exp(x - T.max(x, axis=x.ndim-1, keepdims=True))
    return x / T.sum(x, axis=x.ndim-1, keepdims=True)
    
def add_to_params(params, new_param):
    params.append(new_param)
    return new_param