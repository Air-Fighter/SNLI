import cPickle
import nltk
import numpy as np
import theano

def load_ny_dict(file):
    """
    Load the embedding dict trained by New York Times
    :param file: String, path to the .pickle
    :return: dict, dictionary
    """
    with open(file, 'r') as f:
        dict = cPickle.load(f)

    return dict

def build_vocab(file, initial=False):
    """
    Build the vocabulary of input file.
    :param file:
    :return:a list of words appeared in the context
    """
    if initial:
        vocab = []
        vocab.append('neutral')
        vocab.append('entailment')
        vocab.append('contradiction')
        with open(file, 'r') as f:
            for line in f:
                s1 = line.split('\t')[1]
                s2 = line.split('\t')[2]
                for word in nltk.word_tokenize(s1):
                    if not word in vocab:
                        vocab.append(word)
                for word in nltk.word_tokenize(s2):
                    if not word in vocab:
                        vocab.append(word)

        with open('data/params/vocab.pickle', 'wb') as f:
            cPickle.dump(vocab, f)
    else:
        with open(file, 'rb') as f:
            vocab = cPickle.load(f)

    return vocab

def generate_embedding(vocab, dict, initial=False):
    """
    Generate the embedding matrix and word2vec dictionary from the input
    vocabulary and dictionary.
    :param vocab: vocabulary, list
    :param dict: dictionary, word->embedding vector
    :param initial: indicate whether this is the building process.
    :return: embedding matrix, word2vec
    """

    if initial:
        print "...initializing embedding matrix and word_to_id"
        ematrix = []
        word2id = {}
        i_matrix = 0

        f = open('data/notin.txt', 'w')
        for i in xrange(len(vocab)):
            if vocab[i] in dict.keys():
                ematrix.append(dict[vocab[i]])
                word2id[vocab[i]] = i_matrix
                i_matrix += 1
            else:
                print >>f, vocab[i]
                continue
        f.close()
        word2id['<unk>'] = len(ematrix)
        ematrix.append([1.] * len(dict[vocab[0]]))
        with open('data/params/ematrix.pickle', 'wb') as f:
            cPickle.dump(ematrix, f)
        with open('data/params/word2id.pickle', 'wb') as f:
            cPickle.dump(word2id, f)
    else:
        with open(vocab, 'rb') as f:
            ematrix = cPickle.load(f)
        with open(dict, 'rb') as f:
            word2id = cPickle.load(f)

    ematrix = np.asarray(ematrix, dtype=theano.config.floatX)

    return ematrix, word2id

def pure(file_name, out_file_name):
    """
    Purify the SNLI corpus, filter out the parser trees
    :param file_name: String, the path and name to SNLI corpus
    :param out_file_name: String, the output file
    :return: None
    """
    labels =[]
    prems = []
    hypos = []

    print '...loading into params'
    with open(file_name, 'r') as f:
        for line in f:
            labels.append(line.split('\t')[0])
            prems.append(line.split('\t')[5].lower().replace("\'", ""))
            hypos.append(line.split('\t')[6].lower().replace("\'", ""))
    print '...echoing params to out_file'
    with open(out_file_name, 'w') as f:
        for i in xrange(len(labels)):
            print >>f, labels[i]+'\t'+prems[i]+'\t'+hypos[i]

def file_to_word_ids(file_name, word2id):
    """
    Transform the text file into three lists (labels, prems, hypoes) based on the dictionary word2id
    :param file_name: the file that needs to be transformed
    :param word2id: dictionary, word->id
    :return: three lists labels, prems, hypoes
    """
    labels = []
    prems = []
    hypoes = []

    print '...loading data from ' + file_name

    keys = word2id.keys()

    with open(file_name, 'r') as f:
        for line in f:
            label = line.strip().split('\t')[0]
            prem = line.strip().split('\t')[1]
            hypo = line.strip().split('\t')[2]

            labels.append(word2id[label])

            line_prem = []
            for word in nltk.word_tokenize(prem.strip()):
                if word in keys:
                    line_prem.append(word2id[word])
                else:
                    line_prem.append(word2id['<unk>'])
            prems.append(line_prem)

            line_hypo = []
            for word in nltk.word_tokenize(hypo.strip()):
                if word in keys:
                    line_hypo.append(word2id[word])
                else:
                    line_hypo.append(word2id['<unk>'])
            hypoes.append(line_hypo)

    return labels, prems, hypoes

def prepare_data(labels, prems, hypoes):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.
    """
    # p: a list of premises
    # h: a list of hypothesises
    p_lengths = [len(p) for p in prems]
    h_lengths = [len(h) for h in hypoes]

    n_samples = len(prems)
    p_maxlen = np.max(p_lengths)
    h_maxlen = np.max(h_lengths)

    p = np.zeros((n_samples, p_maxlen)).astype('int32')
    p_mask = np.zeros((n_samples, p_maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(prems):
        p[idx, :p_lengths[idx]] = s
        p_mask[idx, :p_lengths[idx]] = 1.

    h = np.zeros((n_samples, h_maxlen)).astype('int32')
    h_mask = np.zeros((n_samples, h_maxlen)).astype(theano.config.floatX)
    for idx, s in enumerate(hypoes):
        h[idx, :h_lengths[idx]] = s
        h_mask[idx, :h_lengths[idx]] = 1.

    labels = np.asarray(labels, dtype='int32')

    return labels, p, p_mask, h, h_mask

initialized = False
train_labels = []
train_prems = []
train_hypoes = []
test_labels = []
test_prems = []
test_hypoes = []
ematrix = None

def snli_iterator(batch_size, train_text='data/train.txt', test_text='data/test.txt', is_train=True):
    global initialized
    global train_labels
    global train_prems
    global train_hypoes
    global test_labels
    global test_prems
    global test_hypoes
    global ematrix

    if not initialized:
        print '...initializing datasets'
        with open('data/snli.pickle', 'rb') as f:
            datasets = cPickle.load(f)
        train_labels, train_prems, train_hypoes = datasets[0]
        test_labels, test_prems, test_hypoes = datasets[1]

        # following code is initialized datasets
        # ematrix, word2id = generate_embedding(vocab='data/params/ematrix.pickle', dict='data/params/word2id.pickle')
        # train_labels, train_prems, train_hypoes = file_to_word_ids(train_text, word2id)
        # test_labels, test_prems, test_hypoes = file_to_word_ids(test_text, word2id)
        # train_sets = [train_labels, train_prems, train_hypoes]
        # test_sets = [test_labels, test_prems, test_hypoes]
        # print '...saving corpus to data/snli.pickle'
        # with open('data/snli.pickle', 'wb') as f:
        #     cPickle.dump([train_sets, test_sets], f)

        initialized = True

    if is_train:
        batch_num = len(train_labels) / batch_size
        for i in xrange(batch_num):
            yield prepare_data(train_labels[i * batch_size : (i + 1) * batch_size],
                               train_prems[i * batch_size : (i + 1) * batch_size],
                               train_hypoes[i * batch_size : (i + 1) * batch_size])
    else:
        batch_num = len(test_labels) / batch_size
        for i in xrange(batch_num):
            yield prepare_data(test_labels[i * batch_size: (i + 1) * batch_size],
                               test_prems[i * batch_size: (i + 1) * batch_size],
                               test_hypoes[i * batch_size: (i + 1) * batch_size])

if __name__ == '__main__':
    for l, p, p_m, h, h_m in snli_iterator(12):
        print l
        print p
        print p_m
        print h
        print h_m
        exit(0)