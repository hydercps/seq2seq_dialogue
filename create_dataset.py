from sys import argv
import logging
from os import path, makedirs
from codecs import getreader, getwriter

from gensim.models import Word2Vec
from pandas import read_csv
import numpy as np

from data_utils import create_vocabulary, build_embedding_matrix, UNK

logger = logging.getLogger()

TESTSET_RATIO = 0.2
MAX_VOCABULARY_SIZE = 70000


def main(in_dataset, in_result_folder):
    if not path.exists(in_result_folder):
        makedirs(in_result_folder)
    vocabulary_path = path.join(in_result_folder, 'vocab.txt')
    if not path.exists(vocabulary_path):
        vocabulary = create_vocabulary(
            in_dataset.values.flatten(),
            MAX_VOCABULARY_SIZE
        )
        with getwriter('utf-8')(open(vocabulary_path, 'w')) as vocab_out:
            for word in vocabulary:
                print >>vocab_out, word
    else:
        with getreader('utf-8')(open(vocabulary_path)) as vocab_in:
            vocabulary = set([])
            for line in vocab_in:
                vocabulary.add(line.strip())
        logger.info('Skipping vocabulary creation step'.format(vocabulary_path))

    train_path = path.join(in_result_folder, 'train.csv')
    test_path = path.join(in_result_folder, 'test.csv')
    if not path.exists(train_path) or not path.exists(test_path):
        for i in xrange(in_dataset.shape[0]):
            for j in xrange(in_dataset.shape[1]):
                utterance = in_dataset.ix[i, j]
                utterance = ' '.join([
                    token if token in vocabulary else UNK
                    for token in utterance.split()
                ])
                in_dataset.ix[i, j] = utterance

        testset_size = int(TESTSET_RATIO * in_dataset.shape[0])
        test_set = in_dataset.iloc[:testset_size, ]
        train_set = in_dataset.iloc[testset_size:, ]
        test_set.to_csv(
            test_path,
            sep=';',
            header=False,
            index=False,
            encoding='utf-8'
        )
        train_set.to_csv(
            train_path,
            sep=';',
            header=False,
            index=False,
            encoding='utf-8'
        )
    else:
        logger.info('Skipping dataset creating step')
    embeddings_file = path.join(in_result_folder, 'embeddings.npy')
    if not path.exists(embeddings_file):
        w2v = Word2Vec.load_word2vec_format(
            '../word2vec_google_news/GoogleNews-vectors-negative300.bin',
            binary=True
        )
        embedding_matrix = build_embedding_matrix(w2v, vocabulary)
        with open(embeddings_file, 'wb') as embeddings_out:
            np.save(embeddings_out, embedding_matrix)
    else:
        logger.info('Skipping embeddings creating step')


if __name__ == '__main__':
    if len(argv) < 3:
        print 'Usage: create_dataset.py <opensubtitles csv> <result folder>'
        exit()
    opensubtitles_csv, result_folder = argv[1:3]
    dataset = read_csv(
        opensubtitles_csv,
        header=None,
        sep=';',
        encoding='utf-8',
        dtype=str
    )
    dataset = dataset.fillna('')
    main(dataset, result_folder)
