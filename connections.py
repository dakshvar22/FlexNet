__author__ = 'madhumathi'

from deepLearningLibrary.layers import *
from abc import ABCMeta, abstractmethod
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
from layers import *
import random
from exceptions import *

class Connection(object):
    '''
    Abstract class for connections between layers. Not to be instantiated
    '''
    def __init__(self, fromLayer, toLayer, targetNeurons=None,regularization=None,
                 initialization=None,return_sequence=False):

        '''
        :param fromLayer: Input connection layer
        :param toLayer: Outgoing layer
        :return:
        '''
        if targetNeurons is None:
            targetNeurons = toLayer.numOfNeurons
        self.fromLayer = fromLayer
        self.toLayer = toLayer
        self.regularization = regularization
        self.initialization = initialization
        self.output = None
        self.params = []
        self.targetNeurons = targetNeurons
        self.return_sequence = return_sequence

    #### Implement the below methods ####

    # def __new__(cls, *args):
    #     # if cls is Layer:
    #     #     raise NaadiAbstractClassInstantiationError('Connectivity')
    #     # return object.__new__(cls, *args)
    #     pass
    #
    # def allow(self, source, dest):
    #     # Given source and destination neurons, return True/False to indicate if a connection exists between them
    #     # raise NaadiNotImplementedError(inspect.currentframe().f_code.co_name, str(type(self)))
    #     pass


class OneToOneConnection(Connection):

    def __init__(self, fromLayer, toLayer, regularization=None, initialization=None,
                 targetNeurons=None,return_sequence=False):

        super(OneToOneConnection, self).__init__(fromLayer,toLayer,
                                                 targetNeurons,regularization,initialization,return_sequence)

    def __str__(self):
        return str(self.__dict__)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def initializeWeights(self):
        '''
        Function to be called in network.compile() function
        :return:
        '''
        self.w = theano.shared(
                    np.asarray(
                        np.ones(shape= (1,self.fromLayer.numOfNeurons)),
                        dtype=theano.config.floatX), name='w', borrow=True,broadcastable=(True,False))

        # COMMENT THE BELOW LINE IF YOU DO NOT WANT TO LEARN O2O connection WEIGHTS!
        self.params.append(self.w)

    def feedForward(self,miniBatchSize):
        self.output = self.fromLayer.output * self.w

class DenseConnection(Connection):
    def __init__(self, fromLayer, toLayer, regularization=None, initialization=None, targetNeurons=None,return_sequence=False):


        super(DenseConnection, self).__init__(fromLayer,toLayer,
                                              targetNeurons,regularization,initialization,return_sequence)

    def __str__(self):
        return str(self.__dict__)

    def initializeWeights(self):

        # Initialize the weight matrix according to the input and output dimensions(fromLayer and toLayer)
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1.0/self.toLayer.numOfNeurons),
                                 size=(self.fromLayer.numOfNeurons,self.targetNeurons)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(1,self.targetNeurons)),
                       dtype=theano.config.floatX),
            name='b', borrow=True,broadcastable=(True,False))

        self.params = [self.w, self.b]

    def feedForward(self,miniBatchSize):

        # Next Line is when we need to use dropout layer - for each minibatch, each column corresponds to a datapoint, so,
        #we can write a loop where we make around " dropout % " of nodes as 0 and use it. Check what happens when bptt.
        #also ask sir how to tune dropout for each Layer and by default for what ratio of data points and paramters
        # must there be a dropout... can we integrate that ratio in our code by default or throw a suggestion to the user
        #Also, should we also throw other such suggestions to the user such as
        # copyOfLayerOutput = self.fromLayer.output

        self.output = T.dot(self.fromLayer.output,self.w) + self.b

class ConvolutedConnection(Connection):
    def __init__(self, fromLayer, toLayer, regularization, initialization,
                 input_shape, filter_shape, stride_length, zero_padding,return_sequence=False):

        super(ConvolutedConnection, self).__init__(fromLayer,toLayer,regularization,initialization,return_sequence)

        '''
        filter shape - 0 - number of filters, 1 - depth, 2 - height, 3 - width
        input_shape - tuple of length 3 - (the number of input feature maps, the image
        height, and the image width)
        '''
        self.input_shape = input_shape
        self.filter_shape = filter_shape
        self.stride_length = stride_length
        self.zero_padding = zero_padding
        # self.activation_fn = activation_fn
        self.numFilters = filter_shape[0]
        self.n_out = (filter_shape[0]*np.prod(filter_shape[2:]))

    def initializeWeights(self):

        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, size=self.filter_shape),
                dtype=theano.config.floatX),name='w', borrow=True)
        self.b = theano.shared(np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(self.filter_shape[0],)),
                       dtype=theano.config.floatX), name='b', borrow=True)

        self.params = [self.w,self.b]

    def feedForward(self,miniBatchSize):
        '''
        Perform Convolution operation on the output of 'fromLayer'
        Remember output of any layer is always flattened, so first need to reshape it w.r.t to input_shape
        CURRENTLY SUPPORTS ONLY 1-D Convolution and 2-D Convolution
        :param minibatchSize:
        :return:
        '''

        ### Reshape according to 'input_shape'
        self.input = self.fromLayer.output.reshape(self.fromLayer.shape_with_minibatch)

        '''insert minibatchsize value also in the input_shape variable, since that will be the complete shape
        of incoming data'''

        inp = list(self.input_shape)
        inp.insert(0,miniBatchSize)
        self.input_shape = list(inp)

        ### Add zero Pads if any
        if self.zero_padding != 0:
            if len(self.input_shape) == 4:
                zero_padding = T.zeros((self.input_shape[0],self.input_shape[1],
                                        self.input_shape[2] + 2*self.zero_padding,
                                        self.input_shape[3] + 2*self.zero_padding),dtype=theano.config.floatX)
                zero_padding = T.set_subtensor(zero_padding[:,:,
                                               self.zero_padding:self.input_shape[2]+self.zero_padding,
                                               self.zero_padding:self.input_shape[3]+self.zero_padding],
                                               self.input)
                self.input = zero_padding
                input_shape = list(self.input_shape)
                input_shape[2] = input_shape[2] + 2* self.zero_padding
                input_shape[3] = input_shape[3] + 2* self.zero_padding
                self.input_shape = tuple(input_shape)
            elif len(self.input_shape) == 3:
                zero_padding = T.zeros((self.input_shape[0],self.input_shape[1],
                                        self.input_shape[2] + 2*self.zero_padding),dtype=theano.config.floatX)
                zero_padding = T.set_subtensor(zero_padding[:,:,
                                               self.zero_padding:self.input_shape[2]+self.zero_padding],
                                               self.input)
                self.input = zero_padding
                input_shape = list(self.input_shape)
                input_shape[2] = input_shape[2] + 2* self.zero_padding
                self.input_shape = tuple(input_shape)
        conv_out = conv.conv2d(
            input=self.input, filters=self.w, filter_shape=self.filter_shape,
            image_shape=self.input_shape
            ,border_mode="valid",subsample=self.stride_length
        )

        self.output = None
        if len(self.input_shape) == 4:
            self.output = conv_out + self.b.dimshuffle('x', 0, 'x', 'x')
        else:
            self.output = conv_out + self.b.dimshuffle('x', 0, 'x')

        self.output = self.output.reshape(self.toLayer.shape_minibatch_flattened)

class MaxPoolingConnection(Connection):
    def __init__(self, fromLayer, toLayer,poolSize,return_sequence=False):

        super(MaxPoolingConnection, self).__init__(fromLayer,toLayer,return_sequence=return_sequence)
        self.poolSize = poolSize

    def __str__(self):
        return str(self.__dict__)

    def initializeWeights(self):
        self.params = []
        return

    def feedForward(self,miniBatchSize):

        ### Reshape according to 'input_shape'
        self.input = self.fromLayer.output.reshape(self.fromLayer.shape_with_minibatch)
        self.output = downsample.max_pool_2d(input=self.input, ds=self.poolSize, ignore_border=True)
        self.output = self.output.reshape(self.toLayer.shape_minibatch_flattened)

class RecurrentConnection(Connection):
    def __init__(self, fromLayer, toLayer, regularization=None, initialization=None):

        super(RecurrentConnection, self).__init__(fromLayer,toLayer,None,regularization,initialization)

        self.targetNeurons = self.toLayer.numOfNeurons

    def __str__(self):
        return str(self.__dict__)

    def initializeWeights(self):

        # Initialize the weight matrix according to the input and output dimensions(fromLayer and toLayer)
        self.w = theano.shared(
            np.asarray(
                np.random.normal(loc=0.0, scale=np.sqrt(1.0/self.toLayer.numOfNeurons),
                                 size=(self.fromLayer.numOfNeurons,self.targetNeurons)),
                dtype=theano.config.floatX),
            name='w', borrow=True)
        self.b = theano.shared(
            np.asarray(np.random.normal(loc=0.0, scale=1.0, size=(1,self.targetNeurons)),
                       dtype=theano.config.floatX),
            name='b', borrow=True,broadcastable=(True,False))

        self.recurrentHiddenState = self.fromLayer.output
        self.params = [self.w, self.b]

    def feedForward(self,miniBatchSize):

        # print 'Last Layer Shape: ', self.fromLayer.shape
        # self.recurrentState = self.T.dot()
        self.recurrentHiddenOutput = T.dot(self.recurrentHiddenState,self.w) + self.b
        self.outputShape = (miniBatchSize,self.toLayer.numOfNeurons) #### Warning: Not in the case where aggregate_method=concat

    # def getUpdatedHiddenOutput(self,toLayer_hiddenState):
    #     return T.dot(toLayer_hiddenState,self.w_o) + self.b_o

