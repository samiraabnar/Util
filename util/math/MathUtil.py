import numpy as np
import theano.tensor as T


class MathUtil(object):

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def is_float(str):
        try:
            float(str)
            return True
        except ValueError:
            return False

    @staticmethod
    def log_softmax(x):
        xdev = x - x.max(1,keepdims=True)
        return xdev - T.log(T.sum(T.exp(xdev), axis=1, keepdims=True))

    @staticmethod
    def categorical_crossentropy_log_domain(log_predictions, targets):
        return -T.sum(targets * log_predictions, axis=1)


if __name__ == '__main__':
    a = np.array([1,2,3])
    b = MathUtil.sigmoid(a)
    print(b)




