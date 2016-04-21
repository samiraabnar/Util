import theano
import numpy as np


def variable(value, dtype=theano.config.floatX, name=None):
    '''Instantiate a tensor variable'''
    value = np.asarray(value, dtype=dtype)
    return theano.shared(value=value, name=name, strict=False)