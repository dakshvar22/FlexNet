__author__ = 'abhijay'
import theano
import theano.tensor as T
import numpy as np

eps = 1e-9

#output-Prediction of the network
#y - True value
        
def meanSquare(output,y):

    return T.mean(((output - y)**2))

def meanSquareLog(output,y):
    return T.mean(((T.log(output + eps) - T.log(y + eps))**2))

def meanAbsolute(output,y):
    return T.mean((abs(output - y)))

def crossEntropy(output,y):
    return -T.mean((y * T.log(output + eps) + (1-y) * T.log(1 - output - eps))[T.arange(y.shape[0]), y])
        
def negativeLogLikelihood(output,y):
    return -T.mean(T.log(output + eps)[T.arange(y.shape[0]), y])

def kullbackLeiblerDivergence(output, y):
    return T.mean((y * (T.log(y + eps)-T.log(output + eps)))[T.arange(y.shape[0]), y])

def poisson(output,y):
    return T.mean( (output - y * T.log(output + eps))[T.arange(y.shape[0]), y])

def cosine_proximity(output,y):
    return -T.mean((output * y)[T.arange(y.shape[0]), y])



def tester():
    '''y1 = theano.shared(np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[7],[9]],dtype=theano.config.floatX))
    out1 = theano.shared(np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[9],[7]],dtype=theano.config.floatX))

    y2 = theano.shared(np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],dtype=theano.config.floatX))
    out2 = theano.shared(np.asarray([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]],dtype=theano.config.floatX))

#	np.asarray([[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] ]


    out3 = theano.shared(np.asarray([[0.2, 0.05, 0.05,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1],[0.1, 0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1] ]
    ,dtype=theano.config.floatX))

    y3 = theano.shared(np.asarray([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]))

    out4 = theano.shared(np.asarray([[0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1]]
    ,dtype=theano.config.floatX))

    y4 = theano.shared(np.asarray([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]))

    out5 = theano.shared(np.asarray([[0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.9, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.9,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.0,0.0,0.9,0.0,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.0,0.0,0.0,0.9,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.9,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.1],
                                     [0.1, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9],
                                     [0.9, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.9, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.9,0.0,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.9,0.0,0.0,0.0,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.0,0.9,0.0,0.0,0.0,0.0,0.1],
                                     [0.1, 0.0, 0.0,0.0,0.0,0.9,0.0,0.0,0.0,0.0],
                                     [0.0, 0.0, 0.0,0.0,0.0,0.0,0.9,0.0,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.9,0.0,0.1],
                                     [0.0, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.9,0.1],
                                     [0.1, 0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.9]]
    ,dtype=theano.config.floatX))

    y5 = theano.shared(np.asarray([0,1,2,3,4,5,6,7,8,9,0,1,2,3,4,5,6,7,8,9]))


    #a = meanSquareLog(out1,y1)
    a1 = meanSquare(out1,y1)
    a2 = meanSquare(out2,y2)
    print "Mean Square error: " + str(a1.eval())
    print "Mean Square error: " + str(a2.eval())
    print "\n"

    b1 = meanSquareLog(out1,y1)
    b2 = meanSquareLog(out2,y2)
    print "Mean Square Log error: " + str(b1.eval())
    print "Mean Square Log error: " + str(b2.eval())
    print "\n"

    c1 = meanAbsolute(out1,y1)
    c2 = meanAbsolute(out2,y2)
    print "Mean Absolute error: " + str(c1.eval())
    print "Mean Absolute error: " + str(c2.eval())
    print "\n"


    y3 = y3.reshape((20,1))
    # print y3.eval()
    d1 = negativeLogLikelihood(out3,y3)
    #d2 = meanAbsolute(out2,y2)
    #print y3.Print()
    #print out3.Print()

    print "Neg log likelihood error: " + str(d1.eval())
    #print "Mean Absolute error: " + str(c2.eval())
    print "\n"


    y4 = y4.reshape((20,1))
    e1 = crossEntropy(out4,y4)
    f1 = negativeLogLikelihood(out4,y4)
    g1 = kullbackLeiblerDivergence(out4, y4)
    h1 = poisson(out4, y4)
    i1 = cosine_proximity(out4,y4)
    print "negative log likelihood error: " + str(f1.eval())
    print "cross Entropy likelihood error: " + str(e1.eval())
    print "kL divergence error: " + str(g1.eval())
    print "Poisson error: " + str(h1.eval())
    print "Cosine error: " + str(i1.eval())
    #print "Mean Absolute error: " + str(c2.eval())
    print "\n"


    y5 = y5.reshape((20,1))
    out5 = out5.reshape((20,10))
    e2 = crossEntropy(out5,y5)
    f2 = negativeLogLikelihood(out5,y5)
    g2 = kullbackLeiblerDivergence(out5, y5)
    h2 = poisson(out5, y5)
    i2 = cosine_proximity(out5,y5)
    print "negative log likelihood error: " + str(f2.eval())
    print "cross Entropy likelihood error: " + str(e2.eval())
    print "kL divergence error: " + str(g2.eval())
    print "Poisson error: " + str(h2.eval())
    print "Cosine error: " + str(i2.eval())
    #print "Mean Absolute error: " + str(c2.eval())
    print "\n"'''
    y1 = np.asarray([7,1,5,2])
    out1 = np.asarray([[0.1,0.2,0.05,0.05,0.1,0.1,0.1,0.1,0.1,0.1],
                       [0.3,0.4,0.05,0.05,0.05,0.05,0.05,0.05,0.0,0.0],
                       [0.03,0.04,0.5,0.05,0.05,0.5,0.05,0.05,0.0,0.0],
                       [0.03,0.04,0.7,0.7,0.05,0.05,0.05,0.05,0.0,0.0]])
    out2 = np.asarray([[0.05,0.2,0.05,0.05,0.1,0.1,0.1,0.05,0.1,0.1],
                       [0.3,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.0,0.0],
                       [0.3,0.4,0.05,0.05,0.05,0.05,0.05,0.05,0.0,0.0],
                       [0.3,0.4,0.05,0.05,0.05,0.05,0.05,0.05,0.0,0.0]])
    print negativeLogLikelihood(out1,y1).eval()
    print negativeLogLikelihood(out2,y1).eval()



tester()
