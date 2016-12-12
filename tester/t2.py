__author__ = 'daksh'

import numpy as np

from time import time

a = np.random.rand(1000,10)
b = np.random.rand(1000,10)
c = np.random.rand(1000,10)
d = np.random.rand(1000,10)
e = [d,b,c]
# s = a
t1 = time()

for eachArray in e:
    a = np.sum((a,eachArray), axis = 0)

print time() - t1

t2 = time()
e = [a,b,c,d]
allInputs = [x for x in e]
s = np.sum(tuple(allInputs),axis = 0)

print time() - t2