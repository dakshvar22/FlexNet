__author__ = 'daksh'

import theano.tensor as T
from theano import *
import numpy as np
from time import time


w = shared(np.asarray(np.zeros((1000,1000)), np.float32))
print type(w)

a = T.dot(w,np.asarray(np.zeros((1000,1000)), np.float32))
print type(a)
# print shared(T.tanh(w)).get_value().shape
#
# a = time()
# for i in range(10000):
#     s = w.get_value().shape
# print time() - a
#
# a = time()
# for i in range(10000):
#     s = w.shape.eval()
# print time() - a


