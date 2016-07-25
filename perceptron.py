import theano
import numpy as np
import theano.tensor as T

class Perceptron(object):
    def __init__(self, input, in_size=200, hidden_size=400, out_size=1, activation=T.tanh):
        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(1./6),
                high=np.sqrt(1./6),
                size=(in_size, )
            ).astype(theano.config.floatX),
            name='W'
        )

        # input = input.append(np.ones((hidden_size,), dtype=theano.config.floatX))

        self.out = T.tanh(T.dot(self.W, input))
        self.params = [self.W]

