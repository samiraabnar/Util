import theano
import numpy as np
import theano.tensor as T


def variable(value, dtype=theano.config.floatX, name=None):
    '''Instantiate a tensor variable'''
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)

def clip_norm(g , c, n):
    if c > 0:
        g = T.switch(T.ge(n, c), g*c/n,g)
    return g


def clip_norms(gs, c):
    norm = T.sqrt(sum([T.sum(g**2) for g in gs]))
    return [clip_norm(g, c, norm) for g in gs]