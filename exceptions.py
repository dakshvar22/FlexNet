__author__ = 'tarun'


def makeErrorMessage(message):
    """
    :rtype : string
    """
    return 'CoolName Exception: %s' % message

class InputLayerNotDefined(Exception):
    def __init__(self, networkName):
        super(InputLayerNotDefined,self).__init__(makeErrorMessage("Input Layer not defined for network %s" % networkName))


class OutputLayerNotDefined(Exception):
    def __init__(self, networkName):
        super(OutputLayerNotDefined,self).__init__(makeErrorMessage("Output Layer not defined for network %s" % networkName))


class ActivationFunctionNotImplemented(Exception):
    def __init__(self, passFunction):
        super(ActivationFunctionNotImplemented,self).__init__(makeErrorMessage("Activation Function is not implemented %s" % passFunction))


class LossFunctionNotImplemented(Exception):
    def __init__(self, lossFunction):
        super(LossFunctionNotImplemented,self).__init__(makeErrorMessage("Loss Function is not implemented %s" % lossFunction))


class MiniBatchSizeNotInteger(Exception):
    def __init__(self, networkName):
        super(MiniBatchSizeNotInteger,self).__init__(makeErrorMessage("For network %s, the mini batch size is not integer " % networkName))


class SizeMismatch(Exception):
    def __init__(self, toLayerShape, fromLayerShape, message):
        super(SizeMismatch,self).__init__(makeErrorMessage("%s . toLayer shape = %d, fromLayer shape = %d" % (message, toLayerShape, fromLayerShape)))


class AggregateMethodNotDefined(Exception):
    def __init__(self, aggregate_method):
        super(AggregateMethodNotDefined,self).__init__(makeErrorMessage("Aggregate Method is not implemented %s" % aggregate_method))


class DropoutPercentInvalid(Exception):
    def __init__(self, dropout):
        super(DropoutPercentInvalid,self).__init__(makeErrorMessage("Dropout level must be in interval [0, 1]. Dropout mentioned is %d" % dropout))


class ConnectionToItself(Exception):
    def __init__(self):
        super(ConnectionToItself,self).__init__(makeErrorMessage("Can not connect the layer to itself in Dense connections"))


class NetworkNotDAG(Exception):
    def __init__(self):
        super(NetworkNotDAG,self).__init__(makeErrorMessage("Network is not a DAG. It contains a cycle."))


class PoolingNotPossible(Exception):
    def __init__(self):
        super(PoolingNotPossible,self).__init__(makeErrorMessage("Downsampling not possible. Please check your poolsize"))

class ConvolutionNotPossible(Exception):
    def __init__(self):
        super(ConvolutionNotPossible,self).__init__(makeErrorMessage("Convolution between the two layers not possible. Please check the convolution configuration"))