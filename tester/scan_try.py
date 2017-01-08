__author__ = 'daksh'

import theano
import theano.tensor as T
import numpy as np
from theano.ifelse import ifelse
c = 0
# nBatch x sequence_length x featureShape
inp = [[[4, 0],
  [3, 2],
  [2, 0],
  [0, 1]],

 [[1, 1],
  [1, 0],
  [1, 3],
  [1, 4]],

 [[4, 0],
  [1, 4],
  [2, 1],
  [4, 3]]]
# print inp
a = theano.shared(np.asarray(inp,dtype=theano.config.floatX))
print a.shape.eval()
# a = a.dimshuffle((1,0,2))

w = theano.shared(
            np.asarray(
                [[1, 0, 0, 0, 1],[0, 1, 0, 1, 0]]),
            name='w', borrow=True)

w2 = theano.shared(
            np.asarray(
                [[1,1,0,0,1],[0,0,0,0,0]]),
            name='w', borrow=True)
print 'w2',w2.eval()
print 'w',w.eval()


sequence_input = T.tensor3('input')
sequence_input_shuffled = sequence_input.dimshuffle((1,0,2))
# sequence_input_shuffled = theano.printing.Print('this is a very important value')(sequence_input.dimshuffle((1,0,2)))

weights = T.matrix('input')
def step(x):
    # global c
    # c+=1
    # print c.eval()
    return T.dot(x,w)

seqLen = 4
seq = np.repeat([seqLen-1],seqLen)
print seq

def step2(x,idx):
    # print type(idx),type(seq)
    return ifelse(T.lt(idx,seqLen-1),T.dot(x,w),T.dot(x,w2))

final,updates = theano.scan(step2,sequences=[sequence_input_shuffled,T.arange(0,sequence_input_shuffled.shape[0])]
                                 ,outputs_info=None)
# final = step2(sequence_input_shuffled[-1])
# print components.shape.eval()
cost = T.mean(final[-1])
gr = T.grad(cost, w2)
# print 'input',a.eval()
# print 'weights',w.eval()
# print 'result',components.eval()

# print components[-1].eval()
iter = theano.function([], outputs=[gr, final, sequence_input_shuffled], givens={sequence_input:a})
result = iter()
print 'result', result[0], result[1].shape,result[1][-1]
print 'shuffled'
print result[2][-1],result[2].shape
# print components.eval()
print T.arange(0,result[2].shape[0]).eval()
# print inp.shape

# result = iter(a,w).shape.eval()

