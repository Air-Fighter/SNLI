import theano
import theano.tensor as T
import numpy as np
import cPickle

import config
from data_reader import generate_embedding
from data_reader import snli_iterator
from config import options

def run_epoch():
    # define symbolic variables
    l = T.ivector('l')
    p = T.imatrix('p')
    p_mask = T.matrix('p_mask', dtype=theano.config.floatX)
    h = T.imatrix('h')
    h_mask = T.matrix('h_mask', dtype=theano.config.floatX)
    lr = T.scalar(name='lr')

    # build model
    print '...building model'
    np_emb, _ = generate_embedding(config.embedding_param_file, config.word2id_param_file)

    model = options['model'](l, p, p_mask, h, h_mask, np_emb, options['word_size'],
                             options['hidden_size'], options['out_size'],
                             options['use_dropout'], options['drop_p'],
                             options['lstm_mean_pooling'])

    # detector = theano.function(inputs=[l, p, p_mask, h, h_mask], outputs=model.error)
    # predictor = theano.function(inputs=[p, p_mask, h, h_mask], outputs=model.output)
    # for l_, p_, p_mask_, h_, h_mask_ in snli_iterator(batch_size=options['valid_batch_size'], is_train=False):
    #     print l_
    #     print predictor(p_, p_mask_, h_, h_mask_)
    #     print detector(l_, p_, p_mask_, h_, h_mask_)
    #     exit(0)

    cost = model.loss
    grads = T.grad(cost, wrt=list(model.params.values()))
    optimizer = options['optimizer']
    f_grad_shared, f_update = optimizer(lr, model.params, grads, [l, p, p_mask, h, h_mask], cost)

    detector = theano.function(inputs=[l, p, p_mask, h, h_mask], outputs=model.error)

    # load parameters from specified file
    if not options['loaded_params'] is None:
        print '... loading parameters from ' + options['loaded_params']
        file_name = options['param_path'] + model.name + '_hidden' + str(options['hidden_size']) + '_lrate' + \
                    str(options['lrate']) + '_batch' + str(options['batch_size']) + '.pickle'
        with open(file_name, 'rb') as f:
            param_dict = cPickle.load(f)
            for k, v in model.params.items():
                v.set_value(param_dict[k])

    # test the performance of initialized parameters
    errors = []
    for l_, p_, p_mask_, h_, h_mask_ in snli_iterator(batch_size=options['valid_batch_size'], is_train=False):
        error = detector(l_, p_, p_mask_, h_, h_mask_)
        errors.append(error)
    print 'test error of random initialized parameters: ' + str(np.mean(errors) * 100) + '%'

    best_perform = np.inf

    # training model
    print '...training model'
    for i in xrange(options['max_epochs']):
        total_loss = 0.
        idx = 0
        for l_, p_, p_mask_, h_, h_mask_ in snli_iterator(batch_size=options['batch_size'], is_train=True):
            this_cost = f_grad_shared(l_, p_, p_mask_, h_, h_mask_)
            f_update(options['lrate'])
            total_loss += this_cost
            print '\r', 'epoch:', i, ', idx:', idx, ', this_loss:', this_cost,
            idx += 1
        print ', total loss:', total_loss

        # validate model performance when necessary
        if (i + 1) % options['valid_freq'] == 0:
            # test performance on train set
            errors = []
            for l_, p_, p_mask_, h_, h_mask_ in snli_iterator(batch_size=options['valid_batch_size'], is_train=True):
                error = detector(l_, p_, p_mask_, h_, h_mask_)
                errors.append(error)
            print '\ttrain error of epoch ' + str(i) + ': ' + str(np.mean(errors) * 100) + '%'

            # test performance on test set
            errors = []
            for l_, p_, p_mask_, h_, h_mask_ in snli_iterator(batch_size=options['valid_batch_size'], is_train=False):
                error = detector(l_, p_, p_mask_, h_, h_mask_)
                errors.append(error)
            print '\ttest error of epoch ' + str(i) + ': ' + str(np.mean(errors) * 100) + '%',

            # judge whether it's necessary to save the parameters
            save = False
            if np.mean(errors) * 100 < best_perform:
                best_perform = np.mean(errors)
                save = True
                print 'best through: ' + str(best_perform)
            elif best_perform == np.inf:
                print 'best through: inf'
            else:
                print 'best through: ' + str(best_perform)

            # save parameters if need
            if save:
                print '\t...saving parameters'
                file_name = options['param_path'] + model.name + '_hidden' + str(options['hidden_size']) + '_lrate' + \
                            str(options['lrate']) + '_batch' + str(options['batch_size']) + '_epoch' + str(i+1) + \
                            '_perform' + str(float(int(best_perform * 10000)) / 100.) + '.pickle'
                with open(file_name, 'wb') as f:
                    new_dict = {}
                    for k, v in model.params.items():
                        new_dict[k] = v.get_value()
                    cPickle.dump(new_dict, f)

if __name__ == '__main__':
    run_epoch()