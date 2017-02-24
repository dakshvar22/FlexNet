__author__ = 'madhumathi'
import theano.tensor as T

epsilon = 10e-5
''' Set of activation functions '''

def sigmoid(z):
    # Sigmoid activation function in terms of the inputs
    return T.nnet.hard_sigmoid(z)


def tanh(z):
    # Tanh activation function in terms of the inputs
    return T.tanh(z)


def relu(z,alpha=0.0):
    # Relu activation function in terms of the inputs
    return T.switch(z<0.0,0.0,z)
    # return T.nnet.relu(z, alpha)

def softmax(z):
    # expz = T.exp(z)
    # total = T.sum(expz)
    # print total
    # return (expz / total)
    # return T.clip(T.nnet.softmax(z),epsilon,1.0-epsilon)
    return T.nnet.softmax(z)
    # return T.nnet.logsoftmax(z)
    # return T.max(x, axis=axis, keepdims=keepdims)


def passthrough(z):
    # Relu activation function in terms of the inputs
    return (z)



''' Set of gating functions '''

def square(z):
    return T.sqr(z)

def log10(z):
    return T.log10(z)

def log2(z):
    return T.log2(z)
