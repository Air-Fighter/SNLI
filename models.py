import theano
import theano.tensor as T
import numpy as np

from layers import *

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

class LSTM_LR_model(object):
    def __init__(self, l, p, p_mask, h, h_mask, emb, word_size=100, hidden_size=400, out_size=3,
                 use_dropout=True, drop_p=0.5, mean_pooling=False, prefix='model_'):
        self.name = 'SNLI_LSTM_LR'

        # L2-normalize the embedding matrix
        emb_ = np.sqrt(np.sum(emb ** 2, axis=1))
        emb = emb / np.dot(emb_.reshape(-1, 1), np.ones((1, emb.shape[1])))

        self.emb = theano.shared(
            value=np.asarray(emb, dtype=theano.config.floatX),
            name=prefix + 'emb',
            borrow=True
        )

        self.p_embedd_layer = Embedding_layer_uniEmb(
            x=p,
            emb=self.emb,
            word_size=word_size,
            prefix='p_embedd_layer_'
        )

        self.p_lstm_layer = LSTM_layer(
            x=self.p_embedd_layer.output,
            mask=T.transpose(p_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            mean_pooling=mean_pooling,
            prefix='p_lstm_'
        )

        self.h_embedd_layer = Embedding_layer_uniEmb(
            x=h,
            emb=self.emb,
            word_size=word_size,
            prefix='h_embedd_layer_'
        )

        self.h_lstm_layer = LSTM_layer(
            x=self.h_embedd_layer.output,
            mask=T.transpose(h_mask),
            in_size=word_size,
            hidden_size=hidden_size,
            mean_pooling=mean_pooling,
            prefix='h_lstm_'
        )

        if use_dropout:
            self.dropout_layer = Dropout_layer(x=T.concatenate([self.p_lstm_layer.output, self.h_lstm_layer.output],
                                                               axis=1),
                                               p=drop_p)

            self.lr_layer = LogisticRegression(
                x=self.dropout_layer.output,
                y=l,
                in_size=hidden_size * 2,
                out_size=out_size
            )
        else:
            self.lr_layer = LogisticRegression(
                x=T.concatenate([self.p_lstm_layer.output, self.h_lstm_layer.output], axis=1),
                y=l,
                in_size=hidden_size * 2,
                out_size=out_size
            )

        self.output = self.lr_layer.y_d

        self.p_d = self.lr_layer.y_given_x[:, 1]

        self.error = self.lr_layer.error

        self.loss = self.lr_layer.loss

        self.param = {prefix+'emb': self.emb}

        self.params = dict(self.param.items() +
                           self.p_embedd_layer.params.items() +
                           self.p_lstm_layer.params.items() +
                           self.h_embedd_layer.params.items() +
                           self.h_lstm_layer.params.items() +
                           self.lr_layer.params.items()
                           )