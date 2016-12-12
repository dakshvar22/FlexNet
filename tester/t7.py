__author__ = 'daksh'

import theano
import theano.tensor as T
import numpy as np

a = T.zeros(shape=(5,10))

b = theano.shared(np.ones(shape=(5,3)))
c = theano.shared(np.ones(shape=(5,6)))

a = T.set_subtensor(a[:, 0:3], b)
print a.eval()

a = T.set_subtensor(a[:,4:10],c)
print a.eval()

# t = theano.function([np.ones(shape=(5,4)),np.ones(shape=(5,6))],a)
# print t
