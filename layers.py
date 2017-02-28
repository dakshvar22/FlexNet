__author__ = 'madhumathi'


from deepLearningLibrary.activations import *
from deepLearningLibrary.costs import *
from deepLearningLibrary.connections import *
from abc import ABCMeta, abstractmethod
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class Layer(object):
    '''
    Abstract class for layers. Not to be instantiated
    '''
    def __init__(self, shape, passFunction, aggregate_method=None, dropout=None,lossFunction=None,ifOutput = False):
        '''
        :param name: Name for the layer
        :param shape: Shape of the current Layer(num of neurons spatially arranged)
        :param inConnections: list of incoming connection objects
        :param outConnections: list of outgoing connection objects
        :param aggregate_method: function to specify method of concatenating
        :return:
        '''

        self.name = None
        self.shape = shape
        self.inConnections = []
        self.outConnections = []
        self.recurrentInConnections = []
        self.recurrentOutConnections = []
        self.numOfNeurons = 1
        self.ifOutput = ifOutput

        for i in self.shape:
            self.numOfNeurons *= i

        self.aggregate_method = aggregate_method
        self.output = None
        self.lossFunction = self.getLossFunction(lossFunction)
        self.passFunction = self.getPassFunction(passFunction)
        self.dropout = dropout


    def setName(self,name):
        self.name = name

    def addIncomingConnection(self,connection):
        '''
        :param connection: new incoming connection to be added
        :return: None
        '''
        if isinstance(connection,RecurrentConnection):
            self.recurrentInConnections.append(connection)
        else:
            self.inConnections.append(connection)

    def addOutgoingConnection(self, connection):
        '''
        :param connection: new outgoing connection to be added
        :return: None
        '''
        if isinstance(connection,RecurrentConnection):
            self.recurrentOutConnections.append(connection)
        else:
            self.outConnections.append(connection)

    def setNetwork(self,networkName):
        self.network = networkName

    def initializeInputOutput(self,mini_batch_size):

        self.input = T.zeros(shape=(mini_batch_size, self.numOfNeurons))
        self.output = T.zeros(shape=(mini_batch_size, self.numOfNeurons))
        self.hiddenState = T.zeros(shape=(mini_batch_size, self.numOfNeurons))

    def aggregateInput(self):
        ### Handles for for multiple outputs

        if self.aggregate_method == 'sum':
            for connection in self.inConnections:

                '''
                if len(connection.fromLayer.shape) != len(connection.toLayer.shape):
                    raise(SizeMismatch(len(connection.fromLayer.shape), len(connection.toLayer.shape),
                                       "Input and Output Layer dimensions not same."))
                else:
                    for i in range(0, len(connection.fromLayer.shape)):
                        if connection.fromLayer.shape[i] != connection.toLayer.shape[i]:
                            raise(SizeMismatch(len(connection.fromLayer.shape[i]), len(connection.toLayer.shape[i]),
                                               "Input and Output Layer dimensions not same."))
                '''

                self.input = self.input + connection.output

        elif self.aggregate_method == 'concat':
            start_index = 0
            inConnectionSizeSum = 0
            for connection in self.inConnections:
                inConnectionSizeSum += connection.targetNeurons

            if self.numOfNeurons != inConnectionSizeSum:
                raise(SizeMismatch(self.numOfNeurons, inConnectionSizeSum,
                                               "Input and Output Layer dimensions not same."))

            for connection in self.inConnections:
                self.input = T.set_subtensor(self.input[:, start_index:start_index+connection.targetNeurons], connection.output)
                start_index += connection.targetNeurons

        elif self.aggregate_method is None:
            if self.inConnections[0].targetNeurons != self.numOfNeurons:
                raise(SizeMismatch(self.numOfNeurons, self.inConnections[0].targetNeurons,
                                               "Input and Output Layer dimensions not same."))

            self.input += self.inConnections[0].output
        else:
            raise(AggregateMethodNotDefined(self.aggregate_method))

    def computeShapes(self,minibatchSize):
        self.shape_minibatch_flattened = (minibatchSize,self.numOfNeurons)
        self.shape_with_minibatch = list(self.shape)
        self.shape_with_minibatch.insert(0,minibatchSize)
        self.shape_with_minibatch = tuple(self.shape_with_minibatch)

    def addDropout(self):
        if self.dropout is not None:
            if self.dropout < 0. or self.dropout >= 1:
                raise(DropoutPercentInvalid(self.dropout))
            rng = RandomStreams()
            retain_prob = 1. - self.dropout

            random_tensor = rng.binomial(self.numOfNeurons, p=retain_prob, dtype=self.output.dtype)
            random_tensor = T.patternbroadcast(random_tensor, [dim == 1 for dim in self.shape_minibatch_flattened])
            self.output *= random_tensor
            self.output /= retain_prob

    def run(self,minibatchSize):
        '''
        Compute the output of each of the incoming connections and then aggregate all such connection output
        to form the input for this layer, followed by an activation function on this total input
        :return:
        '''
        # print self.name
        ### compute shape Tuples(Hack :( )
        self.computeShapes(minibatchSize)

        self.input = T.zeros(shape=(minibatchSize,self.numOfNeurons))

        ### compute connection outputs (currently flattened outputs)
        for connection in self.inConnections:
            connection.feedForward(minibatchSize)

        # Compute aggregate input from previously calculated connection outputs
        '''
        ADD MORE AGGREGATION FUNCTIONS
        '''
        self.aggregateInput()

        for recurrentConnection in self.recurrentInConnections:
            self.input += recurrentConnection.recurrentHiddenOutput

        '''
        if len(self.recurrentInConnections) > 0:
            # self.hiddenState = self.passFunction(self.input)
            aggregatedRecurrentInput = T.zeros((minibatchSize,self.numOfNeurons))
            for recurrentConnection in self.recurrentInConnections:
                recurrentConnection.fromLayer.hiddenState = self.passFunction(self.input)
                # self.output += recurrentConnection.getUpdatedHiddenOutput(recurrentConnection.fromLayer.hiddenState)
                aggregatedRecurrentInput += recurrentConnection.getUpdatedHiddenOutput(recurrentConnection.fromLayer.hiddenState)
            self.output = self.passFunction(aggregatedRecurrentInput)
        else:
            self.output = self.passFunction(self.input)
        '''
        self.output = theano.printing.Print('Output Val for layer ' + self.name + '=')(self.passFunction(self.input))

        # Add dropout
        if self.dropout > 0.:
            self.addDropout()


    def cost(self, y
             ,predict_y
             , size):
        "Return the log-likelihood cost."
        # Only supports 1D output
        # self.output = self.output.reshape((size, self.numOfNeurons))
        predict_y = predict_y.reshape((size,self.numOfNeurons))
        # return -T.mean(T.log(self.output)[T.arange(size), y])
        # return self.lossFunction(predict_y,y)
        return self.lossFunction(predict_y,y)
        # return self.lossFunction(self.output,y)
        # return -T.mean(y * T.log(self.output) + (1-y) * T.log(1 - self.output))

    def accuracy(self, predict_y,y):
        "Return the accuracy for the mini-batch."
        self.y_out = T.argmax(predict_y, axis=1)
        return T.mean(T.eq(y, self.y_out))

    def getPassFunction(self,passFunction):

        # passFunction is a string input
        activationFunction = None
        if passFunction == "sigmoid":
            activationFunction = sigmoid
        elif passFunction == "tanh":
            activationFunction = tanh
        elif passFunction == "relu":
            activationFunction = relu
        elif passFunction == "softmax":
            activationFunction = softmax
        elif passFunction == "passthrough":
            activationFunction = passthrough
        else:
            raise(ActivationFunctionNotImplemented(passFunction))
        '''
            TODO : ADD MORE PASS FUNCTIONS
        '''
        return activationFunction

    def getLossFunction(self,lossFunction):

        loss = None
        if lossFunction == "meanSquared":
            loss = meanSquare
        elif lossFunction == "meanSquaredLog":
            loss = meanSquareLog
        elif lossFunction == "meanAbsolute":
            loss = meanAbsolute
        elif lossFunction == "crossEntropy":
            loss = crossEntropy
        elif lossFunction == "negativeLogLikelihood":
            loss = negativeLogLikelihood
        elif lossFunction == "klDivergence":
            loss = kullbackLeiblerDivergence
        elif lossFunction == "poisson":
            loss = poisson
        elif lossFunction == "cosineProximity":
            loss = cosine_proximity
        else:
            # pass
            if self.ifOutput:
                raise(LossFunctionNotImplemented(lossFunction))
            else:
                pass
        '''
            TODO : ADD MORE LOSS FUNCTIONS
        '''
        return loss


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

class InputLayer(Layer):

    def __init__(self, inputShape,passFunction="passthrough",aggregate_method=None,
                 lossFunction=None, ifOutput=False,sequence_length=1):
        super(InputLayer,self).__init__(inputShape,passFunction,aggregate_method,lossFunction)
        self.sequence_length = sequence_length

    def firstLayerRun(self, input, minibatchSize):
        ### compute shape Tuples(Hack :( )
        self.computeShapes(minibatchSize)
        self.input = theano.printing.Print('Input Val for layer ' + self.name + '=')(input.reshape(self.shape_minibatch_flattened))
        self.output = theano.printing.Print('Output Val for layer ' + self.name + '=')(self.passFunction(self.input))

class ActivationLayer(Layer):

    def __init__(self, inputShape, passFunction,aggregate_method=None, lossFunction=None, ifOutput=False, dropout=None):

        super(ActivationLayer,self).__init__(inputShape,passFunction,aggregate_method,lossFunction=lossFunction,ifOutput=ifOutput)

        # self.ifOutput = ifOutput  # this is a boolean

class MemoryLayer(Layer):
    # check this!!!
    def __init__(self, inputShape,passFunction,
                 aggregate_method=None, lossFunction=None, ifOutput=False, dropout=None):

        super(MemoryLayer,self).__init__(inputShape,passFunction,aggregate_method,lossFunction=lossFunction,ifOutput=ifOutput)

        # self.ifOutput = ifOutput  # this is a boolean

    def run(self,minibatchSize):

        self.computeShapes(minibatchSize)
        self.input = T.zeros(shape=(minibatchSize,self.numOfNeurons))

        # compute connection outputs (currently flattened outputs)
        for connection in self.inConnections:
            connection.feedForward(minibatchSize)

        # Compute aggregate input from previously calculated connection outputs
        self.aggregateInput()

        for recurrentConnection in self.recurrentInConnections:
            self.input += recurrentConnection.recurrentHiddenOutput

        ''' May be buggy : not sure if self.output will be reinitialized to 0 in each run. Ideally it should not'''
        '''
        if len(self.recurrentInConnections) > 0:
            # self.hiddenState = self.passFunction(self.input)
            aggregatedRecurrentInput = T.zeros((minibatchSize,self.numOfNeurons))
            for recurrentConnection in self.recurrentInConnections:
                recurrentConnection.fromLayer.hiddenState = self.passFunction(self.input)
                # self.output += recurrentConnection.getUpdatedHiddenOutput(recurrentConnection.fromLayer.hiddenState)
                aggregatedRecurrentInput += recurrentConnection.getUpdatedHiddenOutput(recurrentConnection.fromLayer.hiddenState)
            self.output += self.passFunction(aggregatedRecurrentInput)
        else:
            self.output += self.passFunction(self.input)
        '''
        self.output += self.passFunction(self.input)

        if self.dropout > 0.:
            self.addDropout()

        self.y_out = T.argmax(self.output, axis=1)