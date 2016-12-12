__author__ = 'daksh'

import theano
import theano.tensor as T
import numpy as np

a = theano.shared(np.random.normal(loc=0.0,size=(2,3)))

print a.eval()

a = T.switch(a<0,0,a)
print a.eval()

# print t([np.ones((7)), np.ones((5,7))])


