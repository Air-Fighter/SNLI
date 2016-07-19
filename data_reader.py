import cPickle
import nltk

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

def generate_embedding():
    pass

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

if __name__ == '__main__':
    dict_file = 'data/embedding/word2v100.pickle'
    #dict = load_ny_dict(file)
    text_file = 'data/whole.txt'
    vocab_file = 'data/params/vocab.pickle'
    vocab = build_vocab(vocab_file)
    print len(vocab)
    print vocab[0], vocab[1], vocab[2], vocab[6]
    # data_file = 'data/bak/snli_1.0_train.txt'
    # out_file = 'data/train.txt'
    # pure(data_file, out_file)