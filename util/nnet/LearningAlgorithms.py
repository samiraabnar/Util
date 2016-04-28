import theano
import theano.tensor as T
import numpy as np
from Util.util.nnet.theanocore import *
from collections import OrderedDict

import sys
sys.path.append('../../')

from Util.util.nnet.theanocore import *



class LearningAlgorithms(object):


    @staticmethod
    def adam(loss, params, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        """
        ADAM update rules
        Default values are taken from [Kingma2014]
        References:
        [Kingma2014] Kingma, Diederik, and Jimmy Ba.
        "Adam: A Method for Stochastic Optimization."
        arXiv preprint arXiv:1412.6980 (2014).
        :parameters:
            - loss : Theano expression
                specifying loss
            - all_params : list of theano.tensors
                Gradients are calculated w.r.t. tensors in all_parameters
            - learning_Rate : float
            - beta1 : float
                Exponentioal decay rate on 1. moment of gradients
            - beta2 : float
                Exponentioal decay rate on 2. moment of gradients
            - epsilon : float
                For numerical stability
            - gamma: float
                Decay on first moment running average coefficient
            - Returns: list of update rules
        """

        beta1 = variable(value=b1,name="beta1")
        beta2 = variable(value=b2,name="beta2")
        learning_rate = variable(value=lr,name="learning_rate")
        epsilon = variable(value=eps,name="epsilon")

        all_grads = theano.grad(theano.gradient.grad_clip(loss,-1,1), params)
        all_grads = clip_norms(all_grads,0.5)

        t_prev = theano.shared(np.float32(0.))
        updates = OrderedDict()

        t = t_prev + 1.0
        a_t = learning_rate/(1.0-beta1**t)

        for param, g_t in zip(params, all_grads):
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            u_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)

            m_t = beta1*m_prev + (1.0-beta1)*g_t
            u_t = T.maximum(beta2*u_prev, abs(g_t))
            step = a_t*m_t/(u_t + epsilon)

            updates[m_prev] = m_t
            updates[u_prev] = u_t
            updates[param] = param - step

        updates[t_prev] = t
        return updates

