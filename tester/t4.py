__author__ = 'daksh'

import theano
import theano.tensor as T
import numpy as np
from time import time

m = np.asarray(np.random.normal(size=(128,3,100,100)))

x = T.tensor4()
t = T.flatten(x)

r = t.reshape((128,3,100,100))
f = theano.function([x],[t,r])

a = time()
for i in range(1,20):
    f(m)
print time() - a