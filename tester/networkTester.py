__author__ = 'daksh'

import os
# os.environ["THEANO_FLAGS"] = "ldflags = -L/usr/local/lib -lopenblas,device=gpu,floatX=float32,exception_verbosity=high,fastmath = True,root = /usr/local/cuda-7.5,flags=-D_FORCE_INLINES,cnmem=0.85"
os.environ["THEANO_FLAGS"] = "exception_verbosity=high , optimizer=None"
import theano
import theano.tensor as T
import numpy as np
from deepLearningLibrary.network import Network
from deepLearningLibrary.layers import *
from deepLearningLibrary.connections import *
from theano.tensor.nnet import softmax
from theano.tensor.nnet import sigmoid
# import _pickle as cPickle
import cPickle
import pickle
import gzip
import random
import time

def shallow(epochs=5):

    start = time.time()

    # net = Network([
    #     convLayer(input_shape =(mini_batch_size, 1, 28, 28),
    #                       filter_shape=(20, 1, 5, 5),stride_length=(1,1),zero_padding=1,activation_fn=sigmoid
    #               ,poolsize=(2,2)),
    #     FullyConnectedLayer(n_in=20*26*26, n_out=100),
    #     # FullyConnectedLayer(n_in=200, n_out=100),
    #     FullyConnectedLayer(n_in=100, n_out=10,activation_fn=softmax)], mini_batch_size)

    '''
    net = Network('Ho Ja Shuru')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(700,1),passFunction='sigmoid')
    # l5 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',aggregate_method='sum',ifOutput=True,lossFunction="negativeLogLikelihood")

    net.connectDense(l1,l2)
    # net.connectOneToOne(l1,l2)
    # net.connectDense(l2,l2)
    net.connectDense(l1,l3)
    net.connectDense(l2,l4)
    net.connectDense(l3,l4)

    # l5 = ActivationLayer(inputShape=(300,1), aggregate_method="concat")
    # net.connectDense(l2,l5, targetFraction=100)
    # net.connectDense(l3,l5, targetFraction=200)

    net.compile(mini_batch_size)
    #net.loadParams("../data/weights/Ho Ja Shuru_EpochNum_12_accuracy_95.85.pickle")
    # exit(0)
    net.fit(training_data, epochs, 0.1,
        validation_data, test_data)
    '''

    '''
    net2 = Network('Convoluted Baba')
    l1 = InputLayer(inputShape=(1,28,28))
    l2 = ActivationLayer(inputShape=(20,26,26),passFunction='relu')
    l3 = ActivationLayer(inputShape=(300,1),passFunction='sigmoid')
    l6 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid',aggregate_method='sum')
    l4 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l5 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net2.connectConvolution(l1,l2,input_shape = (1,28,28),
                            filter_shape=(20,1,5,5),
                            stride_length=(1,1),zero_padding=1)
    net2.connectDense(l2,l3)
    net2.connectDense(l2,l4)
    net2.connectDense(l3,l6)
    net2.connectDense(l4,l6)
    net2.connectDense(l6,l5)

    # net2.connectDense(l3,l4)
    # net2.connectDense(l4,l5)

    net2.compile(mini_batch_size)

    net2.fit(training_data, epochs, 0.1,
        validation_data, test_data)
    '''

    '''
    net3 = Network('Test O2O')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    # l5 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    l3 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net3.connectDense(l1,l2)
    # net.connectOneToOne(l1,l2)
    # net.connectDense(l2,l2)
    net3.connectOneToOne(l2,l3)
    net3.connectDense(l3,l4)

    # l5 = ActivationLayer(inputShape=(300,1), aggregate_method="concat")
    # net.connectDense(l2,l5, targetFraction=100)
    # net.connectDense(l3,l5, targetFraction=200)

    net3.compile(mini_batch_size)

    net3.fit(training_data, epochs, 0.1,
        validation_data, test_data)
    '''

    '''
    net4 = Network('Testing concat agg method')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    # l5 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid',aggregate_method='concat')
    l5 = ActivationLayer(inputShape=(10,1),passFunction='softmax',
                         ifOutput=True,lossFunction="negativeLogLikelihood")

    net4.connectDense(l1,l2)
    # net.connectOneToOne(l1,l2)
    # net.connectDense(l2,l2)
    net4.connectDense(l1,l3)
    net4.connectDense(l2,l4,targetNeurons=150)
    net4.connectDense(l3,l4,targetNeurons=50)
    net4.connectDense(l4,l5)
    # l5 = ActivationLayer(inputShape=(300,1), aggregate_method="concat")
    # net.connectDense(l2,l5, targetFraction=100)
    # net.connectDense(l3,l5, targetFraction=200)

    net4.compile(mini_batch_size)

    net4.fit(training_data, epochs, 0.1,
        validation_data, test_data)
    '''

    '''
    net6 = Network('Simple Bheja')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    # l5 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    l3 = MemoryLayer(inputShape=(100,1),passFunction='sigmoid')
    # l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net6.connectDense(l1,l2)
    net6.connectDense(l2,l3)
    net6.connectDense(l3,l4)
    # net6.connectRecurrent(l3,l1)
    # l5 = ActivationLayer(inputShape=(300,1), aggregate_method="concat")
    # net.connectDense(l2,l5, targetFraction=100)
    # net.connectDense(l3,l5, targetFraction=200)

    net6.compile(mini_batch_size)

    net6.fit(training_data, epochs, 0.1,
        validation_data, test_data)
    '''

    '''
    net7 = Network('Concat wala Bheja')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(120,1),passFunction='sigmoid')
    l5 = ActivationLayer(inputShape=(80,1),passFunction='sigmoid')
    l3 = MemoryLayer(inputShape=(200,1),passFunction='sigmoid',aggregate_method='concat')
    # l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net7.connectDense(l1,l2)
    net7.connectDense(l2,l3,targetNeurons=160)
    net7.connectDense(l5,l3,targetNeurons=40)
    net7.connectDense(l3,l4)
    net7.connectDense(l1,l5)
    # net6.connectRecurrent(l3,l1)
    # l5 = ActivationLayer(inputShape=(300,1), aggregate_method="concat")
    # net.connectDense(l2,l5, targetFraction=100)
    # net.connectDense(l3,l5, targetFraction=200)

    net7.compile(mini_batch_size)

    net7.fit(training_data, epochs, 0.1,
        validation_data, test_data)
    '''

    '''
    net7 = Network('Test Recurrent')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    # l5 = ActivationLayer(inputShape=(80,1),passFunction='sigmoid')
    l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    # l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net7.connectRecurrent(l3,l3)
    net7.connectDense(l1,l2)
    net7.connectDense(l2,l3)
    net7.connectDense(l3,l4)

    net7.compile(mini_batch_size)

    net7.fit(training_data, epochs, 0.1,validation_data, test_data)
    '''


    '''
    #Does not work with batch size = 15/20 even with sigmoid
    net2 = Network('Keras wala Convoluted Baba(Chalta hua)')
    l1 = InputLayer(inputShape=(1,28,28))
    l2 = ActivationLayer(inputShape=(20,24,24),passFunction='relu')
    l3 = ActivationLayer(inputShape=(20,20,20),passFunction='relu')
    l4 = ActivationLayer(inputShape=(20,10,10),passFunction='passthrough')
    l5 = ActivationLayer(inputShape=(128,1),passFunction='relu')
    l6 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")
    net2.connectConvolution(l1,l2,input_shape = (1,28,28),
                            filter_shape=(20,1,5,5),
                            stride_length=(1,1),zero_padding=0)
    net2.connectConvolution(l2,l3,input_shape=(20,24,24),
                            filter_shape=(20, 20, 5, 5),
                            stride_length=(1,1),zero_padding=0)

    net2.connectMaxPool(l3,l4,poolSize=(2,2))
    net2.connectDense(l4,l5)
    net2.connectDense(l5,l6)

    net2.compile(mini_batch_size)

    net2.fit(training_data, epochs, 0.1,
        validation_data, test_data)

    '''
    '''
    net2 = Network('Keras wala Convoluted Baba')
    l1 = InputLayer(inputShape=(1,28,28))
    l2 = ActivationLayer(inputShape=(32,26,26),passFunction='relu')
    l3 = ActivationLayer(inputShape=(32,24,24),passFunction='relu')
    l4 = ActivationLayer(inputShape=(32,12,12),passFunction='passthrough',dropout=0.25)
    l5 = ActivationLayer(inputShape=(128,1),passFunction='sigmoid',dropout=0.5) #used 'relu' originally
    l6 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")
    net2.connectConvolution(l1,l2,input_shape = (1,28,28),
                            filter_shape=(32,1,3,3),
                            stride_length=(1,1),zero_padding=0)
    net2.connectConvolution(l2,l3,input_shape = (32,26,26),
                            filter_shape=(32,32,3,3),
                            stride_length=(1,1),zero_padding=0)

    net2.connectMaxPool(l3,l4,poolSize=(2,2))
    net2.connectDense(l4,l5)
    net2.connectDense(l5,l6)

    net2.compile(mini_batch_size)

    net2.fit(training_data, epochs, 0.2,
        validation_data, test_data,lmbda=0.3)
    '''
    '''
    net7 = Network('Dropout')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid', dropout = 0.5)
    l5 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid', dropout = 0.5)
    #l3 = MemoryLayer(inputShape=(6,1),passFunction='sigmoid',aggregate_method='concat')
    # l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net7.connectDense(l1,l2)
    #net7.connectDense(l2,l3,targetNeurons=3)
    #net7.connectDense(l5,l3,targetNeurons=3)
    net7.connectDense(l2,l5)
    net7.connectDense(l5,l4)
    # net6.connectRecurrent(l3,l1)
    # l5 = ActivationLayer(inputShape=(300,1), aggregate_method="concat")
    # net.connectDense(l2,l5, targetFraction=100)
    # net.connectDense(l3,l5, targetFraction=200)

    net7.compile(mini_batch_size)

    net7.fit(training_data, epochs, 0.1,validation_data, test_data)
    '''

    '''
    net7 = Network('RecurrentCheck')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    #l3 = MemoryLayer(inputShape=(6,1),passFunction='sigmoid',aggregate_method='concat')
    # l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net7.connectDense(l1,l2)
    net7.connectDense(l2,l3)
    net7.connectDense(l3,l4)
    net7.connectRecurrent(l3,l2)
    net7.connectRecurrent(l2,l4)
    net7.connectRecurrent(l2,l2)
    # l5 = ActivationLayer(inputShape=(300,1), aggregate_method="concat")
    # net.connectDense(l2,l5, targetFraction=100)
    # net.connectDense(l3,l5, targetFraction=200)

    net7.compile(mini_batch_size)
    #net7.loadParams("../data/weights/RecurrentCheck_EpochNum_4_accuracy_94.61.pickle")
    net7.fit(training_data, epochs, 0.1,
        validation_data, test_data)
    '''

    '''net = Network('Ho Ja Shuru')
    l1 = InputLayer(inputShape = (784,1))
    l2 = ActivationLayer(inputShape=(700,1),passFunction='sigmoid')
    l5 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net.connectDense(l1,l2)
    # net.connectOneToOne(l1,l2)
    # net.connectDense(l2,l2)
    net.connectDense(l2,l5)
    net.connectDense(l5,l3)
    net.connectDense(l3,l4)

    net.compile(mini_batch_size)
    # net.loadParams("../data/weights/Ho Ja Shuru_EpochNum_14_accuracy_92.2475961538")
    net.fit(training_data, epochs, 0.1, validation_data, test_data)
    end = time.time()
    print end - start

    return net'''

    '''
    net = Network('Check Errors')
    l1 = InputLayer(inputShape = (784,1),sequence_length=4)
    l2 = ActivationLayer(inputShape=(700,1),passFunction='sigmoid')
    l5 = ActivationLayer(inputShape=(200,1),passFunction='sigmoid')
    l3 = ActivationLayer(inputShape=(100,1),passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,1),passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net.connectDense(l1,l2,return_sequence=True)
    # net.connectOneToOne(l1,l2)
    # net.connectDense(l2,l2)
    net.connectDense(l2,l5,return_sequence=True)
    net.connectDense(l5,l3,return_sequence=False)
    net.connectDense(l3,l4,return_sequence=False)

    net.compile(mini_batch_size)
    # net.loadParams("../data/weights/Ho Ja Shuru_EpochNum_14_accuracy_92.2475961538")
    net.fit(training_data, epochs, 0.1, validation_data, test_data)
    '''

    '''
    net = Network('Sequences')
    l1 = InputLayer(inputShape = (196,), sequence_length=4)
    l2 = ActivationLayer(inputShape=(300,), passFunction='sigmoid')
    l5 = ActivationLayer(inputShape=(150,), passFunction='sigmoid')
    l3 = ActivationLayer(inputShape=(100,), passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,), passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net.connectDense(l1,l2,return_sequence=True)
    # net.connectOneToOne(l1,l2)
    # net.connectDense(l2,l2)
    # net.connectRecurrent(l5,l2)
    net.connectDense(l2,l5,return_sequence=True)
    net.connectDense(l5,l3,return_sequence=False)
    net.connectDense(l3,l4,return_sequence=False)

    net.compile(mini_batch_size)
    # net.loadParams("../data/weights/Ho Ja Shuru_EpochNum_14_accuracy_92.2475961538")
    net.fit(training_data, epochs, 0.1, validation_data, test_data)
    '''

    net = Network('Check Multiple Output')
    l1 = InputLayer(inputShape = (196,), sequence_length=4)
    l2 = ActivationLayer(inputShape=(300,), passFunction='sigmoid')
    l5 = ActivationLayer(inputShape=(150,), passFunction='sigmoid')
    l6 = ActivationLayer(inputShape=(10,), passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")
    l3 = ActivationLayer(inputShape=(100,), passFunction='sigmoid')
    l4 = ActivationLayer(inputShape=(10,), passFunction='softmax',ifOutput=True,lossFunction="negativeLogLikelihood")

    net.connectDense(l1,l2,return_sequence=True)
    # net.connectOneToOne(l1,l2)
    # net.connectDense(l2,l2)
    net.connectRecurrent(l5,l2)
    net.connectDense(l5,l6)
    net.connectDense(l2,l5,return_sequence=True)
    net.connectDense(l5,l3)
    net.connectDense(l3,l4)

    net.compile(mini_batch_size)
    # net.loadParams("../data/weights/Ho Ja Shuru_EpochNum_14_accuracy_92.2475961538")
    net.fit(training_data, epochs, 0.1, validation_data, test_data)


    end = time.time()
    print end - start

    return net



#### Load the MNIST data
def load_data_shared(filename="../data/mnist.pkl.gz"):
    f = gzip.open(filename, 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    print training_data[0][0].shape,training_data[1].shape
    f.close()
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.
        """
        shared_x = theano.shared(
            np.asarray(data[0].reshape(data[0].shape[0],data[0].shape[1]/196,data[0].shape[1]/4), dtype=theano.config.floatX), borrow=True)
            # np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]

mini_batch_size = 20
training_data, validation_data, test_data = load_data_shared()

net = shallow(epochs=60)
