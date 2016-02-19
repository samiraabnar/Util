import numpy as np
class MathUtil(object):

    def softmax(x):
        xt = np.exp(x - np.max(x))
        return xt / np.sum(xt)