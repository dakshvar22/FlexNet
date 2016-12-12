__author__ = 'daksh'

import theano
import theano.tensor as T
import numpy as np

# nBatch x sequence_length x featureShape
inp = np.random.randint(5, size=(3, 4, 2))
print inp
a = theano.shared(np.asarray(inp,dtype=theano.config.floatX))
# a = a.dimshuffle((1,0,2))

w = theano.shared(
            np.asarray(
                np.random.randint(2,size=(2,5))),
            name='w', borrow=True)

sequence_input = T.tensor3('input')
sequence_input_shuffled = sequence_input.dimshuffle((1,0,2))
weights = T.matrix('input')
l = []
def step(x):
    return T.dot(x,w)

components,updates = theano.scan(step,sequences=sequence_input_shuffled[:-1]
                                 # ,non_sequences=l
                                 ,outputs_info=None)
# print components.shape.eval()

# print 'input',a.eval()
# print 'weights',w.eval()
# print 'result',components.eval()

# print components[-1].eval()
iter = theano.function([],outputs=components,givens={sequence_input:a})
result = iter()
print 'result',result,result.shape

result = iter()
print 'result',result,result.shape

result = iter()
print 'result',result,result.shape

# print inp.shape

# result = iter(a,w).shape.eval()

