import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict

from options import options

class Forward_Layer(object):
    def __init__(self, input, hidden_size=options['hidden_size'], word_dim=options['word_dim']*2):
        """
        Calculate attention value to every word in the other sentence
        :param input: concatenation of the premise and hypothesis
        :param hidden_size: the size of hidden layer
        :param word_dim: the dimension of input word embedding
        :return self.out: a single value in (-1, 1)
        """
        self.W = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(1. / 6),
                high=np.sqrt(1. / 6),
                size=(word_dim, hidden_size)
            ).astype(theano.config.floatX),
            name='W'
        )

        self.U = theano.shared(
            value=np.random.uniform(
                low=-np.sqrt(1. / 6),
                high=np.sqrt(1. / 6),
                size=(hidden_size, )
            ).astype(theano.config.floatX),
            name='U'
        )

        # input = input.append(np.ones((hidden_size,), dtype=theano.config.floatX))

        self.out = T.tanh(T.dot(T.dot(self.W, input), self.U))
        self.params = OrderedDict({
            'word_attend_W':self.W,
            'word_attend_U':self.U
        })

class Model(object):
    def __init__(self, prems, hypoes):
        pass

def run():
    batch_size = 16
    prems = np.random.randint(low=0, high=99, size=(batch_size, 5), dtype='int32')
    hypoes = np.random.randint(low=0, high=99, size=(batch_size, 3), dtype='int32')
    labels = np.random.randint(low=0, high=3, size=(batch_size,), dtype='int32')
    print prems
    print hypoes
    print labels

    ematrix = np.random.uniform(low=-1, high=1, size=(100, 100)).astype(theano.config.floatX)

    t_prems = T.imatrix('p')
    t_hypoes = T.imatrix('h')
    t_ematrix = theano.shared(ematrix, 't_ematrix')

    r_prems = T.repeat(t_prems, 3, axis= 1)
    r_hypoes = T.concatenate([t_hypoes]* 5, axis=1)

    batch_prems = t_ematrix[r_prems]
    batch_hypoes = t_ematrix[r_hypoes]

    batch_prem_hypo = T.concatenate((batch_prems, batch_hypoes), axis=2)

    get_b_prems = theano.function(inputs=[t_prems], outputs=batch_prems)
    get_r_prems = theano.function(inputs=[t_prems], outputs=r_prems)
    get_b_hypoes = theano.function(inputs=[t_hypoes], outputs=batch_hypoes)
    get_r_hypoes = theano.function(inputs=[t_hypoes], outputs=r_hypoes)
    get_b_ph = theano.function(inputs=[t_prems, t_hypoes], outputs=batch_prem_hypo)

    # print get_b_prems(prems)
    print get_r_prems(prems)
    print get_r_hypoes(hypoes)

    print get_b_prems(prems).shape
    print get_b_hypoes(hypoes).shape

    print get_b_ph(prems, hypoes).shape

    W = theano.shared(
        value=np.random.uniform(
            low=-np.sqrt(1. / 6),
            high=np.sqrt(1. / 6),
            size=(200, 400)
        ).astype(theano.config.floatX),
        name='W'
    )

    U = theano.shared(
        value=np.random.uniform(
            low=-np.sqrt(1. / 6),
            high=np.sqrt(1. / 6),
            size=(400,)
        ).astype(theano.config.floatX),
        name='U'
    )

    result = T.dot(T.dot(batch_prem_hypo, W), U)

    get_result = theano.function(inputs=[t_prems, t_hypoes], outputs=result)
    print get_result(prems, hypoes).shape

if __name__ == '__main__':
    run()