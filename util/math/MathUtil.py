import numpy as np


class MathUtil(object):

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / np.sum(e_x)

    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))


if __name__ == '__main__':
    a = np.array([1,2,3])
    b = MathUtil.sigmoid(a)
    print(b)


