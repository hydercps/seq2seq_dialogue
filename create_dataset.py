from itertools import count
from sys import argv
import logging
from os import path, makedirs
from codecs import getreader, getwriter

from gensim.models import Word2Vec
from pandas import read_csv, DataFrame
import numpy as np
from json import load

from data_utils import (
    create_vocabulary,
    build_embedding_matrix,
    UNK_ID,
    find_bucket,
    pad_sequence
)

logger = logging.getLogger()

TESTSET_RATIO = 0.2
MAX_VOCABULARY_SIZE = 70000


def main(in_dataset, in_result_folder, in_config):
    if not path.exists(in_result_folder):
        makedirs(in_result_folder)
    vocabulary_path = path.join(in_result_folder, 'vocab.txt')
    if not path.exists(vocabulary_path):
        vocabulary_list = create_vocabulary(
            in_dataset.values.flatten(),
            MAX_VOCABULARY_SIZE
        )
        vocabulary = {
            token: token_index
            for token_index, token in enumerate(vocabulary_list)
        }
        with getwriter('utf-8')(open(vocabulary_path, 'w')) as vocab_out:
            for word in vocabulary_list:
                print >>vocab_out, word
    else:
        with getreader('utf-8')(open(vocabulary_path)) as vocab_in:
            vocabulary = {}
            for line, line_index in zip(vocab_in, count()):
                vocabulary[line.strip()] = line_index
        logger.info('Skipping vocabulary creation step'.format(vocabulary_path))

    train_path = path.join(in_result_folder, 'train.csv')
    test_path = path.join(in_result_folder, 'test.csv')
    if not path.exists(train_path) or not path.exists(test_path):
        for i in xrange(in_dataset.shape[0]):
            modified_row = []
            utterances = in_dataset.iloc[i, ]
            for utterance in utterances:
                utterance_ids = [
                    str(vocabulary[token]) if token in vocabulary else str(UNK_ID)
                    for token in utterance.split()
                ]
                modified_row.append(' '.join(utterance_ids))
            in_dataset.ix[i, ] = modified_row

        testset_size = int(TESTSET_RATIO * in_dataset.shape[0])
        test_set = in_dataset.iloc[:testset_size, ]
        train_set = in_dataset.iloc[testset_size:, ]
        '''test_set.to_csv(
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
        )'''
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
    for component_name, input_buckets in pad_and_bucket_dataset(train_set, vocabulary, in_config).iteritems():
        for bucket_index, bucket in enumerate(input_buckets):
            file_path = path.join(
                in_result_folder,
                'train_{}_{}.npy'.format(component_name, bucket_index)
            )
            np.save(file_path, bucket)
    for component_name, input_buckets in pad_and_bucket_dataset(test_set, vocabulary, in_config).iteritems():
        for bucket_index, bucket in enumerate(input_buckets):
            file_path = path.join(
                in_result_folder,
                'test_{}_{}.npy'.format(component_name, bucket_index)
            )
            np.save(file_path, bucket)


def pad_and_bucket_dataset(in_dataset_csv, in_vocabulary, in_config):
    reverse_vocabulary = {
        value: key
        for key, value in in_vocabulary.iteritems()
    }
    BUCKETS = in_config['buckets']
    bucketed_encoder_inputs = [[] for _ in BUCKETS]
    bucketed_decoder_inputs = [[] for _ in BUCKETS]
    for i in xrange(in_dataset_csv.shape[0]):
        encoder_input, decoder_input = in_dataset_csv.iloc[i, ]
        encoder_input_ids = [int(token_id) for token_id in encoder_input.split()]
        decoder_input_ids = [int(token_id) for token_id in decoder_input.split()]
        bucket_id = find_bucket(
            len(encoder_input_ids),
            len(decoder_input_ids),
            BUCKETS
        )
        if bucket_id is None:
            logger.warn('Couldn\'t find bucket for a pair')
            continue
        encoder_pad_length, decoder_pad_length = BUCKETS[bucket_id]
        padded_encoder_input = pad_sequence(
            encoder_input_ids,
            encoder_pad_length
        )
        padded_decoder_input = pad_sequence(
            decoder_input_ids,
            decoder_pad_length,
            reverse_vocabulary,
            to_onehot=True
        )
        bucketed_encoder_inputs[bucket_id].append(padded_encoder_input)
        bucketed_decoder_inputs[bucket_id].append(padded_decoder_input)

    return {
        'encoder': [np.asarray(input_bucket) for input_bucket in bucketed_encoder_inputs],
        'decoder': [np.asarray(input_bucket) for input_bucket in bucketed_decoder_inputs],
    }





if __name__ == '__main__':
    if len(argv) < 3:
        print 'Usage: create_dataset.py <opensubtitles csv> <config json> <result folder>'
        exit()
    opensubtitles_csv, config_file, result_folder = argv[1:4]
    with getreader('utf-8')(open(config_file)) as config_in:
        config = load(config_in)
    dataset = read_csv(
        opensubtitles_csv,
        header=None,
        sep=';',
        encoding='utf-8',
        dtype=str
    )
    dataset = dataset.fillna('')
    main(dataset, result_folder, config)
