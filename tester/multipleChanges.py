import theano
import theano.tensor as T

a = T.scalar()
b = T.scalar()

c = a
c = c+b

f = theano.function([a,b],c)

print f(2,3)