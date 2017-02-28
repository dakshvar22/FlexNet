import theano.tensor as T
from deepLearningLibrary.activations import *

from deepLearningLibrary.layers import *
from deepLearningLibrary.connections import *
from deepLearningLibrary.toposort import *
from pprint import pprint
import math
import cPickle
import time
from theano.ifelse import ifelse
import theano.d3viz as d3v

class Network(object):

    def __init__(self, name):
        '''

        :param name: Network Name to be initialised
        :return: None
        '''
        self.layers = []    #List of Layers
        self.optimizer = None   #Optimizer Type (Currently only SGD)
        self.connections = []   #List of Connections
        self.name = name    #Initialize name of Network
        self.outputLayers = []

    def addLayer(self, layer):
        '''
        :param layer: Layer object to be added to network
        :return:
        '''
        self.layers.append(layer)
        layer.setNetwork(self.name) #define the name in the model(supreme) class and assign it.

    def updateLayersInNetwork(self,fromLayer,toLayer):
        '''
        :param fromLayer: Layer from which connection originates
        :param toLayer:  Layer at which connection terminates
        :return:
        '''
        if(fromLayer not in self.layers):
            if isinstance(fromLayer,InputLayer):
                self.inputLayer = fromLayer
            self.addLayer(fromLayer)

        if(toLayer not in self.layers):
            self.addLayer(toLayer)

    def updateConnectionsInNetwork(self,fromLayer,toLayer,newConnection):
        '''
        :param fromLayer: Layer from which connection originates
        :param toLayer:  Layer at which connection terminates
        :param newConnection: New Connection object to be added to network
        :return:
        '''
        fromLayer.addOutgoingConnection(newConnection)
        toLayer.addIncomingConnection(newConnection)
        self.connections.append(newConnection)

    def connectOneToOne(self, fromLayer, toLayer,
                        regularization = None, initialization = None,return_sequence=False):
        '''
        :param fromLayer: Layer from which connection originates
        :param toLayer:  Layer at which connection terminates
        :param regularization:  Regularization scheme to be used for these set of weights
        :param initialization:  Initialization scheme to be used for initialization of weights
        :return:
        '''

        if fromLayer == toLayer:
            raise(ConnectionToItself())

        # Add Layers to self.layers if layer does not exist already
        self.updateLayersInNetwork(fromLayer,toLayer)

        # Create a new Connection object and update connections of network
        c = OneToOneConnection(fromLayer, toLayer, regularization, initialization,return_sequence)
        self.updateConnectionsInNetwork(fromLayer,toLayer,c)

    def connectDense(self, fromLayer, toLayer, regularization = None,
                     initialization = None,targetNeurons=None,return_sequence=False):
        '''
        :param fromLayer: Layer from which connection originates
        :param toLayer:  Layer at which connection terminates
        :param regularization:  Regularization scheme to be used for these set of weights
        :param initialization:  Initialization scheme to be used for initialization of weights
        :param targetNeurons:   Number of Neurons to be to be used on the target layer
        :return:
        '''

        if fromLayer == toLayer:#check how to compare 2 layers
            raise(ConnectionToItself())

        if toLayer.aggregate_method == 'sum' and targetNeurons is not None and toLayer.numOfNeurons != targetNeurons:
            raise(SizeMismatch(targetNeurons, toLayer.numOfNeurons,
                                               "Input and Output Layer dimensions not same."))

        # Add Layers to self.layers if layer does not exist already
        self.updateLayersInNetwork(fromLayer,toLayer)

        # Create a new Connection object and update connections of network
        c = DenseConnection(fromLayer, toLayer, regularization, initialization,
                            targetNeurons=targetNeurons,return_sequence=return_sequence)
        self.updateConnectionsInNetwork(fromLayer,toLayer,c)

    def connectConvolution(self, fromLayer, toLayer, input_shape, filter_shape,
                           stride_length, zero_padding, regularization = None,
                           initialization = None,return_sequence=False):
        '''
        :param fromLayer: Layer from which connection originates
        :param toLayer:  Layer at which connection terminates
        :param input_shape: Input Shape of this connection
        :param filter_shape:    Filter Shape to be used for convolution(Kernel Shape)
        :param stride_length:   Stride length by which kernel should be moved in each iteration
        :param zero_padding:    Zero padding to be added if any to the input
        :param regularization:  Regularization scheme to be used for these set of weights
        :param initialization:  Initialization scheme to be used for initialization of weights
        :return:
        '''

        if fromLayer == toLayer:#check how to compare 2 layers
            raise(ConnectionToItself())

        # Check if Convolution is possible (2D Convolution only)
        if toLayer.shape[-1] != ((fromLayer.shape[-1] - filter_shape[-1] + 2*zero_padding)/stride_length[-1] + 1) or \
                        toLayer.shape[-2] != ((fromLayer.shape[-2] - filter_shape[-2] + 2*zero_padding)/stride_length[-2] + 1) or \
                        fromLayer.shape[-3] != filter_shape[-3] or toLayer.shape[-3] != filter_shape[-4]:
            raise(ConvolutionNotPossible())

        # Add Layers to self.layers if layer does not exist already
        self.updateLayersInNetwork(fromLayer,toLayer)

        # Create a new Connection object and update connections of network
        c = ConvolutedConnection(fromLayer, toLayer, regularization,
                           initialization, input_shape, filter_shape,
                           stride_length, zero_padding,return_sequence)
        self.updateConnectionsInNetwork(fromLayer,toLayer,c)

    def connectRecurrent(self,fromLayer, toLayer,
                         regularization = None, initialization = None):
        '''
        :param fromLayer: Layer from which connection originates
        :param toLayer:  Layer at which connection terminates
        :param regularization:  Regularization scheme to be used for these set of weights
        :param initialization:  Initialization scheme to be used for initialization of weights
        :return:
        '''
        # Add Layers to self.layers if layer does not exist already
        self.updateLayersInNetwork(fromLayer,toLayer)

        # Create a new Connection object and update connections of network
        c = RecurrentConnection(fromLayer, toLayer, regularization, initialization)
        self.updateConnectionsInNetwork(fromLayer,toLayer,c)

    def connectMaxPool(self,fromLayer,toLayer,poolSize,return_sequence=False):

        '''
        :param fromLayer: Layer from which connection originates
        :param toLayer:  Layer at which connection terminates
        :param poolSize: sampling frequency
        '''

        if fromLayer == toLayer:#check how to compare 2 layers
            raise(ConnectionToItself())

        # Check if Max Pooling is possible(2D only)
        if (not (fromLayer.shape[-1]/float(poolSize[-1])).is_integer()) or (not (fromLayer.shape[-1]/float(poolSize[-1])).is_integer()) or \
                        toLayer.shape[-1] != fromLayer.shape[-1]/poolSize[-1] or toLayer.shape[-2] != fromLayer.shape[-2]/poolSize[-2]:
            raise(PoolingNotPossible())

        # Add Layers to self.layers if layer does not exist already
        self.updateLayersInNetwork(fromLayer,toLayer)

        # Create a new Connection object and update connections of network
        c = MaxPoolingConnection(fromLayer, toLayer, poolSize,return_sequence)
        self.updateConnectionsInNetwork(fromLayer, toLayer, c)

    def checkErrors(self, miniBatchSize):

        if not isinstance(miniBatchSize,int):
            raise(MiniBatchSizeNotInteger(self.name))
        isInput = False
        isOutput = False

        for layer in self.layers:
            if(isinstance(layer,InputLayer)):
                isInput = True
            else:
                if(layer.ifOutput == True):
                    isOutput = True

        if not isInput:
            raise(InputLayerNotDefined(self.name))
        if not isOutput:
            raise(OutputLayerNotDefined(self.name))

    def getTimeVariantLayers(self):
        '''
        If any layer has all the incoming connections as return_sequence = false, then the output of that layer has
        to be computed at the last time step only. Otherwise, if even one incoming connection has return_sequence = true
        then the output for that layer has to be computed at each time step
        :return: set of layers which operate on timesteps
        '''
        timeLayers = []
        for layer in self.layers:
            timeLayerFound = False
            for connection in layer.inConnections:
                if connection.return_sequence:
                    timeLayerFound = True
                    break
            if isinstance(layer,InputLayer):
                if layer.sequence_length > 1:
                    timeLayerFound = True
            if timeLayerFound:
                timeLayers.append(layer)
        return timeLayers

    def timeVariationalStep(self,x):
        # Define the feedforward equations for each layer
        for layer in self.timeVariantLayers:
            if isinstance(layer,InputLayer):
                layer.firstLayerRun(x,self.mini_batch_size)
            else:
                # Defining the input and output for each layer
                layer.run(self.mini_batch_size)

        ''' this is wrong - return the output of all those layers which are around the
        corner in time varying layers(see example)'''
        # return self.timeVariantLayers[-1].output
        return [layer.output for layer in self.outputLayers]

    def step(self,x):
        # Define the feedforward equations for each layer
        print 'inside step'
        for layer in self.layers:
            if isinstance(layer,InputLayer):
                print "inputLayer also present"
                layer.firstLayerRun(x,self.mini_batch_size)
            else:
                # Defining the input and output for each layer
                # print layer.name
                layer.run(self.mini_batch_size)

        return [layer.output for layer in self.outputLayers]

    def step2(self,x,idx):

        # print self.sequence_length-1
        t = T.switch(T.lt(idx,self.sequence_length-1),self.timeVariationalStep(x),self.step(x))
        #t = theano.ifelse.ifelse(T.lt(idx,self.sequence_length-1),self.timeVariationalStep(x),self.step(x))
        # Update the hidden states of the recurrent connection with the new updated output of fromLayer and run
        # feedForward so that output variable of that connection gets updated

        ''' Check if this correct over here for time varying case'''
        for connection in self.connections:
            if isinstance(connection,RecurrentConnection):
                connection.recurrentHiddenState = connection.fromLayer.output
                connection.feedForward(self.mini_batch_size)

        # return [(layer.output,layer.name) for layer in self.outputLayers]
        return [layer.output for layer in self.outputLayers]

    def compile(self, mini_batch_size):
        '''
        :param mini_batch_size: batch size to be used for this network
        :return:
        '''
        # Assign names to layers
        self.checkErrors(mini_batch_size)
        self.namingLayers()

        # Construct a DAG out of the network
        g = constructGraph(self.layers)

        # Apply toposort and get the updated order of layers in which feedforward should run
        self.layers = topological(g)

        # Get time variant layers
        self.timeVariantLayers = self.getTimeVariantLayers()
        for i in self.timeVariantLayers:
            print i.name
        print 'printed timeVariantLayers'

        self.mini_batch_size = mini_batch_size

        # Initialize input and output variables(symbolic) for each layer
        for layer in self.layers:
            layer.initializeInputOutput(mini_batch_size)
            # Also check the sequence length which is default to 1.
            # If different sequence length in multiple input layers, then this code will have to be changed
            if isinstance(layer,InputLayer):
                self.inputLayer = layer
                #print layer.input, layer.output
                self.sequence_length = layer.sequence_length
            # Assigning the output layer object of the network to the appropriate layer
            if layer.ifOutput:
                self.outputLayers.append(layer)
        #
        # for i in self.layers:
        #     print i.input,i.output

        # Symbolic theano variable for the input matrix to the network
        inputLen = len(self.inputLayer.shape)
        # print self.sequence_length,'hell'
        if (self.sequence_length == 1):
            self.x = T.TensorType(theano.config.floatX,(False,)*(inputLen + 1))()
        else:
            self.x = T.TensorType(theano.config.floatX,(False,)*(inputLen + 2),name='inputMatrix')()
        print self.x.type()

        # print self.x.ndim

        # For each connection initialize the weights(parameters) of the connection
        for connection in self.connections:
            connection.initializeWeights()

            # if the connection is of type Recurrent, then run feedForward once to initialize the hidden state of
            # that connection

            # This no longer needed since we have initialized recurrentHiddenOutput variable to zeros
            if isinstance(connection,RecurrentConnection):
                connection.feedForward(mini_batch_size)

        # if self.sequence_length > 1:
        #     shuffledList = [x for x in range(inputLen+2)]
        #     shuffledList[0],shuffledList[1] = 1,0
        #
        #     self.x_shuffled = self.x.dimshuffle(tuple(shuffledList))
        #     self.output_r, scan_updates = theano.scan(self.timeVariationalStep,sequences=self.x_shuffled[:-1]
        #                          # ,non_sequences=self.timeVariantLayers
        #                          ,outputs_info=None)
        #     # self.output,scan_updates = theano.scan(self.step,sequences=self.x_shuffled[-1:]
        #     #                                        # ,non_sequences=self.layers
        #     #                                        ,outputs_info=None)
        #     self.output = self.step(self.x_shuffled[-1:],self.output_r)[0]
        # else:
        #     self.output = self.step(self.x)[0]
        if self.sequence_length > 1:
            shuffledList = [x for x in range(inputLen+2)]
            shuffledList[0],shuffledList[1] = 1,0
            x_shuffled = self.x.dimshuffle(tuple(shuffledList))

            output, self.scan_updates = theano.scan(self.step2,
                                               sequences=[x_shuffled,T.arange(x_shuffled.shape[0])]
                                                 # ,non_sequences=self.timeVariantLayers
                                                 ,outputs_info=None,
                                                    # truncate_gradient = 0
                                                # ,n_steps=self.sequence_length-1
                                                 )
            self.output = output[-1]
        else:
            self.output = self.step(self.x)
        # print type(self.scan_updates)
        # timeFunctionCall = theano.function([],[output],updates=scan_updates)
        # out = timeFunctionCall()
        # self.output = self.step(self.x_shuffled[-1:])

        # Aggregate all parameters of the network(Used for updation during backpropagation)
        self.params = [param for connection in self.connections for param in connection.params]

        # self.output = self.outputLayer.output

    def computeCost(self,idx,output):
        return self.outputLayers[idx].cost(output,self.mini_batch_size)

    def fit(self, training_data, epochs, eta,
            validation_data, test_data, lmbda=0.0):
        '''
        :param training_data:   Data to be trained on
        :param epochs:  Number of epochs the network should be run for
        :param eta: Learning Rate to be used
        :param validation_data: Validation Data for parameter tuning of the network
        :param test_data:   Data for which predictions have to be made
        :param lmbda:   Regularization Constant
        :return:
        '''
        """Train the network using mini-batch stochastic gradient descent."""

        # self.mini_batch_size = mini_batch_size
        training_x, training_y = training_data
        validation_x, validation_y = validation_data
        test_x, test_y = test_data

        # compute number of minibatches for training, validation and testing
        num_training_batches = int(self.size(training_data)/self.mini_batch_size)
        num_validation_batches = int(self.size(validation_data)/self.mini_batch_size)
        num_test_batches = int(self.size(test_data)/self.mini_batch_size)

        print('batch sizes')
        print(num_training_batches,num_validation_batches,num_test_batches)

        self.y = T.ivector('ground truth')

        # define the (regularized) cost function, symbolic gradients, and updates
        # costs,updates = theano.scan(self.computeCost,
        #                             sequences=T.arange(len(self.outputLayers)),
        #                             non_sequences=self.y)
        # cost = T.sum([layer.cost(self.y,self.mini_batch_size) for layer in self.outputLayers])
        # cost = T.sum(costs)
        cost = 0.0
        # print type(self.output)
        # cost = self.outputLayers[0].cost(self.y,self.mini_batch_size) + self.outputLayers[1].cost(self.y,self.mini_batch_size)

        # if isinstance(self.output,list):
        #     for ind,layer in enumerate(self.outputLayers):
        #         cost += layer.cost(self.y,self.output[ind],self.mini_batch_size)
        # else:
        #     print 'came here'
        #     cost += self.outputLayers[0].cost(self.y,self.output,self.mini_batch_size)

        for ind,layer in enumerate(self.outputLayers):
            cost += layer.cost(self.y,self.output[ind],self.mini_batch_size)

        # cost = self.outputLayers[0].cost(self.y,self.output[0],self.mini_batch_size) \
        #        + self.outputLayers[1].cost(self.y,self.output[1],self.mini_batch_size)

        cost_printed = theano.printing.Print('Cost val = ')(cost)
        # for output,layerName in self.output:
        #     cost += self.layer_by_name[layerName].cost(self.y,self.mini_batch_size)
        # print self.output
        # cost += self.outputLayers[0].lossFunction(self.output[0],self.y) + \
        #        self.outputLayers[1].lossFunction(self.output[1],self.y)
        # cost = self.outputLayers[1].lossFunction(self.outputLayers[1].output,self.y)
        # cost = negativeLogLikelihood(self.output,self.y)

        # y_out = T.argmax(self.output,axis=1)
        # accuracy = T.mean(T.eq(y_out,self.y))
        # theano.printing.debugprint(cost)
        theano.printing.pydotprint(cost,outfile='pics/graph_correct.png',format='png',var_with_name_simple=True)
        theano.printing.pydotprint(self.output,outfile='pics/graph_output.png',format='png',var_with_name_simple=True)
        theano.printing.pydotprint(self.outputLayers[0].output,outfile='pics/graph_output_outputLayer.png',format='png',var_with_name_simple=True)

        # d3v.d3viz(self.outputLayers[0].output,'pics/output_interactive.html')

        accuracy = 0.0

        # if isinstance(self.output,list):
        #     for ind,layer in enumerate(self.outputLayers):
        #         accuracy += layer.accuracy(self.output[ind],self.y)
        #     accuracy /= len(self.outputLayers)
        # else:
        #     accuracy = self.outputLayers[0].accuracy(self.output,self.y)

        for ind,layer in enumerate(self.outputLayers):
            accuracy += layer.accuracy(self.output[ind],self.y)
        accuracy /= len(self.outputLayers)

        # accuracy = self.outputLayers[0].accuracy(self.y)

        grads = T.grad(cost, self.params, disconnected_inputs='warn')
        updates = [(param, param-eta*grad)
                   for param, grad in zip(self.params, grads)]
        # print type(updates)

        # define functions to train a mini-batch, and to compute the
        # accuracy in validation and test mini-batches.
        i = T.lscalar() # mini-batch index

        train_mb = theano.function(
            [i],
            # [cost,self.layers[-1].output,self.layers[-1].input,self.connections[-1].w],
            [cost],
            # mode=theano.compile.MonitorMode(
            #             pre_func=inspect_inputs,
            #             post_func=inspect_outputs),
            updates=updates,
            givens={
                self.x:
                training_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size] ,
                self.y:
                training_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            }
            ,on_unused_input='ignore',
            name='training_func'
        )
        # theano.printing.debugprint(train_mb)
        # theano.printing.pydotprint(train_mb,outfile='graph.png',format='png')

        validate_mb_accuracy = theano.function(
            [i],
            # [self.outputLayers[-1].accuracy(self.y)],
            # [cost],
            [accuracy],
            givens={
                self.x:
                validation_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                validation_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            },on_unused_input='ignore'
        ,name='validation accuracy function')

        test_mb_accuracy = theano.function(
            [i],
            # self.outputLayers[-1].accuracy(self.y),
            # [cost],
            [accuracy],
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size],
                self.y:
                test_y[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            },on_unused_input='ignore',
        name='test accuracy function'
        )
        '''
        test_mb_predictions = theano.function(
            [i], # self.layers[-1].accuracy(self.y),
            [layer.output for layer in self.outputLayers],
            # [self.layers[-1].output
            #     ,
            #       self.layers[-2].output
            #       ],
            givens={
                self.x:
                test_x[i*self.mini_batch_size: (i+1)*self.mini_batch_size]
            },on_unused_input='ignore'
        ,name='test predictions function')
        '''

        # Do the actual training
        best_validation_accuracy = 0.0
        '''
        if(savingFrequency == 0):
            #lets keep saving and overwriting after every 20% percent of epochs
            savingFrequency = int(0.2 * epochs)
            if(savingFrequency == 0):
                savingFrequency = 1
        '''
        for epoch in range(epochs):
            tic = time.time()
            for minibatch_index in range(num_training_batches):
                iteration = num_training_batches*epoch+minibatch_index
                if iteration % 1000 == 0:
                    print("Training mini-batch number {0}".format(iteration))
                cost_ij = train_mb(minibatch_index)
                print 'minibatch ended'
                if (iteration+1) % num_training_batches == 0:
                    validation_accuracy = np.mean(
                        [validate_mb_accuracy(j) for j in range(num_validation_batches)])
                    print("Epoch {0}: validation accuracy {1:.2%}".format(
                        epoch, validation_accuracy))
                    print("Corresponding Loss : ",cost_ij)

                    if validation_accuracy > best_validation_accuracy:
                        print("This is the best validation accuracy to date.")
                        best_validation_accuracy = validation_accuracy
                        best_iteration = iteration


                        # # Save Model for future reference
                        # for eachConnection in self.connections:
                        #     connectionName = eachConnection.toLayer.name + "+" + eachConnection.fromLayer.name
                        #     print connectionName
                        #     fileName = "../data/weights/" + self.name + "_EpochNum_" + str(epoch) + "_accuracy_" + str(best_validation_accuracy*100) + connectionName + ".pickle"
                        #     #dictToSave[connectionName] = eachConnection.params
                        #     #dictToSave[connectionName] = [param.get_value() for param in eachConnection.params]
                        #     saveList = eachConnection.params
                        #     print fileName
                        #     with open(fileName, 'wb') as handle:
                        #         cPickle.dump(saveList, handle, protocol=cPickle.HIGHEST_PROTOCOL)
                        #
                        # #To Load
                        # #with open(filename.pickle, 'rb') as handle:
                        # #    unserialized_data = pickle.load(handle)
                        #

                        if test_data:
                            test_accuracy = np.mean(
                                [test_mb_accuracy(j) for j in range(num_test_batches)])
                            print('The corresponding test accuracy is {0:.2%}'.format(
                                test_accuracy))


            print time.time() - tic
        print("Finished training network.")
        print("Best validation accuracy of {0:.2%} obtained at iteration {1}".format(
            best_validation_accuracy, best_iteration))
        print("Corresponding test accuracy of {0:.2%}".format(test_accuracy))

    '''
    Everything below this is work in progress
    '''
    def loadParams(self,filePath):
        #Loading the params :

        for eachConnection in self.connections:
            connectionName = eachConnection.toLayer.name + "+" + eachConnection.fromLayer.name
            fileName = filePath + connectionName + ".pickle"
            with open(fileName, 'rb') as handle:
                dataLoaded = cPickle.load(handle)
            try:
                eachConnection.params = dataLoaded #What about convoluted connection
                #eachConnection.params = [theano.shared(param.eval()) for param in dataLoaded[connectionName]]#10%
                #eachConnection.params = [theano.shared(param).eval() for param in dataLoaded[connectionName]]#works
                #eachConnection.params = [theano.shared(param) for param in dataLoaded[connectionName]]
                print "yes"
                if isinstance(eachConnection,OneToOneConnection):
                    print len(eachConnection.params)
                    eachConnection.w = eachConnection.params[0]
                elif isinstance(eachConnection,DenseConnection):
                    print len(eachConnection.params)
                    eachConnection.w = eachConnection.params[0]
                    eachConnection.b = eachConnection.params[1]
                    #eachConnection.w = theano.shared(eachConnection.params[0],name='w', borrow=True)
                    #eachConnection.b = theano.shared(eachConnection.params[1],name='b', borrow=True)
                elif isinstance(eachConnection,RecurrentConnection):
                    print len(eachConnection.params)
                    eachConnection.w_h = eachConnection.params[0]
                    eachConnection.w_o = eachConnection.params[1]
                    eachConnection.b_h = eachConnection.params[2]
                    eachConnection.b_o = eachConnection.params[3]
                    #self.params = [self.w_h, self.w_o, self.b_h,self.b_o]
                print connectionName, eachConnection.params
            except Exception, e:
                print repr(e)
                # raise "Different connections than the one you are trying to Load. Please check your network."



        # print('loaded weight',self.connections[-1].w.eval())
        self.params = [param for connection in self.connections for param in connection.params]


    def loadParams1(self,fileName):
        #Loading the params :
        print "entered"
        if(fileName.split(".")[-1] != "pickle"):
            raise "The file should be a pickle file"
        with open(fileName, 'rb') as handle:
            dataLoaded = cPickle.load(handle)

        #print len(dataLoaded)

        for eachConnection in self.connections:
            connectionName = eachConnection.toLayer.name + "+" + eachConnection.fromLayer.name
            try:
                #eachConnection.params = dataLoaded[connectionName] #What about convoluted connection
                #eachConnection.params = [theano.shared(param.eval(),.get_value()) for param in dataLoaded[connectionName]]#10%
                #eachConnection.params = [theano.shared(param).eval() for param in dataLoaded[connectionName]]#works
                #eachConnection.params = [theano.shared(param) for param in dataLoaded[connectionName]]
                tempParams = []
                for eachParam in dataLoaded[connectionName]:
                    print type(eachParam)#<class 'theano.sandbox.cuda.var.CudaNdarraySharedVariable'>

                #exit(0)
                if isinstance(eachConnection,OneToOneConnection):
                    print len(eachConnection.params)
                    eachConnection.w = theano.shared(dataLoaded[connectionName][0],name='w', borrow=True)
                    eachConnection.params = [eachConnection.w]
                elif isinstance(eachConnection,DenseConnection):
                    print len(eachConnection.params)
                    eachConnection.w = theano.shared(dataLoaded[connectionName][0],name='w', borrow=True)
                    eachConnection.b = theano.shared(dataLoaded[connectionName][1],name='b', borrow=True)
                    eachConnection.params = [eachConnection.w,eachConnection.b]
                elif isinstance(eachConnection,RecurrentConnection):
                    print len(eachConnection.params)
                    eachConnection.w_h = theano.shared(dataLoaded[connectionName][0],name='w_h', borrow=True)
                    eachConnection.w_o = theano.shared(dataLoaded[connectionName][1],name='w_o', borrow=True)
                    eachConnection.b_h = theano.shared(dataLoaded[connectionName][2],name='b_h', borrow=True)
                    eachConnection.b_o = theano.shared(dataLoaded[connectionName][3],name='b_o', borrow=True)
                    eachConnection.params = [eachConnection.w_h,eachConnection.w_o,eachConnection.b_h,eachConnection.b_o]

                print connectionName, eachConnection.params
            except Exception, e:
                print repr(e)
                # raise "Different connections than the one you are trying to Load. Please check your network."
        '''

        for eachConnection in self.connections:
            connectionName = eachConnection.toLayer.name + "+" + eachConnection.fromLayer.name
            try:
                #eachConnection.params = dataLoaded[connectionName] #What about convoluted connection
                #eachConnection.params = [theano.shared(param.eval()) for param in dataLoaded[connectionName]]#10%
                #eachConnection.params = [theano.shared(param).eval() for param in dataLoaded[connectionName]]#works
                eachConnection.params = [theano.shared(param) for param in dataLoaded[connectionName]]
                print "yes"
                if isinstance(eachConnection,OneToOneConnection):
                    print len(eachConnection.params)
                    eachConnection.w = eachConnection.params[0]
                elif isinstance(eachConnection,DenseConnection):
                    print len(eachConnection.params)
                    eachConnection.w = eachConnection.params[0]
                    eachConnection.b = eachConnection.params[1]
                    #eachConnection.w = theano.shared(eachConnection.params[0],name='w', borrow=True)
                    #eachConnection.b = theano.shared(eachConnection.params[1],name='b', borrow=True)
                elif isinstance(eachConnection,RecurrentConnection):
                    print len(eachConnection.params)
                    eachConnection.w_h = eachConnection.params[0]
                    eachConnection.w_o = eachConnection.params[1]
                    eachConnection.b_h = eachConnection.params[2]
                    eachConnection.b_o = eachConnection.params[3]
                    #self.params = [self.w_h, self.w_o, self.b_h,self.b_o]
                print connectionName, eachConnection.params
            except Exception, e:
                print repr(e)
                # raise "Different connections than the one you are trying to Load. Please check your network."

        '''

        # print('loaded weight',self.connections[-1].w.eval())
        self.params = [param for connection in self.connections for param in connection.params]

    def size(self,data):
        "Return the size of the dataset `data`."
        return data[0].get_value(borrow=True).shape[0]

    def printConnections(self):
        print 'Connections Layer Wise : \n======================'

        for i in self.layers:
            print 'For Layer with shape :', i.shape
            print 'Incoming Connections :\n-----------------'
            for j in i.inConnections:
                print j.fromLayer.shape , j.toLayer.shape
            print '------------------------'

            print 'Outgoing Connections :\n-----------------'
            for j in i.outConnections:
                print j.fromLayer.shape , j.toLayer.shape
            print '------------------------'
    '''
    def namingLayers(self):
        #output names as Layer(<class 'deepLearningLibrary.layers.ActivationLayer'>) 1
        self.layer_by_name = dict()
        # Builds a dictionary of the layers by name.
        for i, layer in enumerate(self.layers):
            # Also generates names for each of the layers, if not given already.
            t = type(layer)
            if not layer.name:
                layer.setName('Layer(%s) %d' %(t,i))

            # Raise an exception if there is a name clash
            if self.layer_by_name.has_key(layer.name):
                raise "Layer Name Clash"
            else:
                self.layer_by_name[layer.name] = layer
    '''
    def namingLayers(self):
        self.layer_by_name = dict()
        # Builds a dictionary of the layers by name.
        for i, layer in enumerate(self.layers):
            # Also generates names for each of the layers, if not given already.
            if not layer.name:
                layer.setName('Layer%d%s' %(i+1,str(layer.shape)))

            # Raise an exception if there is a name clash
            if self.layer_by_name.has_key(layer.name):
                raise "Layer Name Clash"
            else:
                self.layer_by_name[layer.name] = layer

    def transform(self, data):
        #transform the dataset if needed
        pass

def inspect_inputs(i, node, fn):
    print(i, node, "input(s) value(s):", [input[0] for input in fn.inputs])

def inspect_outputs(i, node, fn):
    print(" output(s) value(s):", [output[0] for output in fn.outputs])

