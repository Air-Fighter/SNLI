import cPickle
import nltk
import numpy as np

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
        ematrix.append([1.] * len(vocab[0]))
        word2id['<unk>'] = len(ematrix)
        with open('data/params/ematrix.pickle', 'wb') as f:
            cPickle.dump(ematrix, f)
        with open('data/params/word2id.pickle', 'wb') as f:
            cPickle.dump(word2id, f)
    else:
        with open(vocab, 'rb') as f:
            ematrix = cPickle.load(f)
        with open(dict, 'rb') as f:
            word2id = cPickle.load(f)

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
    labels = []
    prems = []
    hypoes = []

    with open(file_name, 'r') as f:
        for line in f:
            label = line.strip().split('\t')[0]
            prem = line.strip().split('\t')[1]
            hypo = line.strip().split('\t')[2]

            labels.append(word2id[label])

            line_prem = []
            for word in nltk.word_tokenize(prem.strip()):
                if word in word2id.keys():
                    line_prem.append(word2id[word])
                else:
                    line_prem.append(word2id['<unk>'])
            prems.append(line_prem)

            line_hypo = []
            for word in nltk.word_tokenize(hypo.strip()):
                if word in word2id.keys():
                    line_hypo.append(word2id[word])
                else:
                    line_hypo.append(word2id['<unk>'])
            hypoes.append(line_hypo)

    labels = np.asarray(labels, dtype=int)
    prems = np.asarray(prems, dtype=int)
    hypoes = np.asarray(hypoes, dtype=int)
    return labels, prems, hypoes

def data_iterator(file_name):
    pass

if __name__ == '__main__':
    # data_file = 'data/bak/snli_1.0_test.txt'
    # out_file = 'data/test.txt'
    # pure(data_file, out_file)
    # text_file = 'data/all.txt'
    # vocab = build_vocab(text_file, initial=True)
    ##########################################################################

    # dict_file = 'data/embedding/word2v100.pickle'
    # dict = load_ny_dict(dict_file)
    vocab_file = 'data/params/vocab.pickle'
    vocab = build_vocab(vocab_file)
    print len(vocab)

    ematrix_file ='data/params/ematrix.pickle'
    word2id_file = 'data/params/word2id.pickle'
    ematrix, word2id = generate_embedding(ematrix_file, word2id_file)

    test_text = 'data/test.txt'
    test_set = file_to_word_ids(test_text, word2id)
    print test_set[0]
    print test_set[1]
    print test_set[2]
