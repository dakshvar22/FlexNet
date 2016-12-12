__author__ = 'abhijay'
import theano.tensor as T
import theano
import numpy as np
eps = 1e-9
#y_pred-Prediction of the network
#y_true - True value

def binaryAccuracy(y_true,y_pred):
    '''Calculates the mean accuracy rate across all predictions for binary
    classification problems.
    '''	
    return T.mean(T.eq(y_true, (y_pred)))
    
def categoricalAccuracy(y_true, y_pred):
    '''Calculates the mean accuracy rate across all predictions for
    multiclass classification problems.
    '''
    return T.mean(T.eq(T.argmax(y_true, axis=-1),T.argmax(y_pred, axis=-1)))
    
    
    
def meanSquaredError(y_true, y_pred):
    '''Calculates the mean squared error (mse) rate
    between predicted and target values.
    '''
    return T.mean((y_pred - y_true)**2)

def meanAbsoluteError(y_true, y_pred):
    '''Calculates the mean absolute error (mae) rate
    between predicted and target values.
    '''
    return T.mean(abs(y_pred - y_true))
    
  
def meanAbsolutePercentageError(y_true, y_pred):
    '''Calculates the mean absolute percentage error (mape) rate
    between predicted and target values.
    '''
    diff = abs((y_true - y_pred) / abs(y_true))
    return 100. * T.mean(diff)

def meanSquaredLogarithmicError(y_true, y_pred):
    '''Calculates the mean squared logarithmic error (msle) rate
    between predicted and target values.
    '''
    
    return T.mean((T.log(y_pred + eps) - T.log(y_true + eps))**2)
    
    
def kullbackLeiblerDivergence(y_true, y_pred):
    '''Calculates the Kullback-Leibler (KL) divergence between prediction
    and target values.
    '''
    return T.sum(y_true * T.log((y_true / y_pred+eps)+eps), axis=-1)


def poisson(y_true, y_pred):
    '''Calculates the poisson function over prediction and target values.
    '''
    return T.mean(y_pred - y_true * T.log(y_pred + eps))


def cosineProximity(y_true, y_pred):
    '''Calculates the cosine similarity between the prediction and target
    values.
    '''
    return -T.mean(y_true * y_pred)	
    
    
def matthewsCorrelation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = T.round(T.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = T.round(T.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = T.sum(y_pos * y_pred_pos)
    tn = T.sum(y_neg * y_pred_neg)

    fp = T.sum(y_neg * y_pred_pos)
    fn = T.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = T.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + eps)


def precision(y_true, y_pred):
    '''Calculates the precision, a metric for multi-label classification of
    how many selected items are relevant.
    '''
    true_positives = T.sum(T.round(T.clip(y_true * y_pred, 0, 1)))
    predicted_positives = T.sum(T.round(T.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + eps)
    return precision
    
    
def recall(y_true, y_pred):
    '''Calculates the recall, a metric for multi-label classification of
    how many relevant items are selected.
    '''
    true_positives = T.sum(T.round(T.clip(y_true * y_pred, 0, 1)))
    possible_positives = T.sum(T.round(T.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + eps)
    return recall



def fBetaScore(y_true, y_pred, beta=1):
    
   

        if beta < 0:
            raise ValueError('The lowest choosable beta is zero (only precision).')




        # If there are no true positives, fix the F score at 0 like sklearn.
        if (T.sum(T.round(T.clip(y_true, 0, 1))) == 0):
            print "no true positives found"
            return 0


        p = precision(y_true, y_pred)
        r = recall(y_true, y_pred)
        bb = beta ** 2
        fbeta_score = (1 + bb) * (p * r) / ((bb * p + r ))
        return fbeta_score

    
def fmeasure(y_true, y_pred):
    '''Calculates the f-measure, the harmonic mean of precision and recall.
    '''
    return fBetaScore(y_true, y_pred, beta=1)
   
   
   
#def tester():
    #y1 = theano.shared(np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[7],[9]],dtype=theano.config.floatX))
    #out1 = theano.shared(np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[9],[7]],dtype=theano.config.floatX))

    #y2 = theano.shared(np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],dtype=theano.config.floatX))
    #out2 = theano.shared(np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],dtype=theano.config.floatX))


    #y3 = theano.shared(np.asarray([0,1,0,1,0,0,0,0,0,0],dtype=theano.config.floatX))
    #out3 = theano.shared(np.asarray([1,1,1,1,1,1,1,1,1,1],dtype=theano.config.floatX))

##	np.asarray([[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] ]




    ##a = meanSquareLog(out1,y1)
    #a1 = binaryAccuracy(out1,y1)
    #a2 = binaryAccuracy(out2,y2)
    #print "Mean Square error: " + str(a1.eval())
    #print "Mean Square error: " + str(a2.eval())
    #print "\n"

    #b1 = categoricalAccuracy(out1,y1)
    #b2 =categoricalAccuracy(out2,y2)
    #print "Mean Square Log error: " + str(b1.eval())
    #print "Mean Square Log error: " + str(b2.eval())
    #print "\n"

    #c1 = meanSquaredError(out1,y1)
    #c2 = meanSquaredError(out2,y2)
    #print "Mean Absolute error: " + str(c1.eval())
    #print "Mean Absolute error: " + str(c2.eval())
    #print "\n"

    #d1 = kullbackLeiblerDivergence(out1,y1)
    #d2 = kullbackLeiblerDivergence(out2,y2)
    #print "Mean Absolute error: " + str(d1.eval())
    #print "Mean Absolute error: " + str(d2.eval())
    #print "\n"



    #e1 = matthewsCorrelation(y1, out1)
    #e2 = matthewsCorrelation(y2, out2)
    #print "Mean Absolute error: " + str(e1.eval())
    #print "Mean Absolute error: " + str(e2.eval())
    #print "\n"


    #f1 = precision(y3,out3)
    #f2 = recall(y3,out3)
    #print "Precision: " + str(f1.eval())
    #print "Recall: " + str(f2.eval())
    #print "\n"

    #g1 = fBetaScore(y3,out3,beta=1)
    #g2 = fBetaScore(y3,out3,beta=1)
    #print "F Beta score: " + str(g1.eval())
    #print "F Beta score: " +str(g2.eval())
    #print "\n"

#tester()	   
