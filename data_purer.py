def pure(file_name, out_file_name):
    labels =[]
    prems = []
    hypos = []

    print '...loading into params'
    with open(file_name, 'r') as f:
        for line in f:
            labels.append(line.split('\t')[0])
            prems.append(line.split('\t')[5])
            hypos.append(line.split('\t')[6])
    print '...echoing params to out_file'
    with open(out_file_name, 'w') as f:
        for i in xrange(len(labels)):
            print >>f, labels[i]+'\t'+prems[i]+'\t'+hypos[i]

if __name__ == '__main__':
    file = 'data/snli_1.0_train.txt'
    out = 'data/train.txt'
    pure(file, out)