__author__ = 'daksh'

from theano import *
import theano
import theano.tensor as T

x = T.zeros(shape=(100,))
y = T.ones(shape=(2,))
print x.eval()
print y.eval()

x = T.set_subtensor(x[2:4],y)
print x.eval()

