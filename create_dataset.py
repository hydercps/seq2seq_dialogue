from itertools import count
from sys import argv
import logging
from os import path, makedirs
from codecs import getreader, getwriter

from gensim.models import Word2Vec
from pandas import read_csv
import numpy as np
from json import load

from data_utils import (
    create_vocabulary,
    build_embedding_matrix,
    find_bucket,
    pad_sequence,
    collect_bucket_stats,
    utterance_to_ids,
    GO_ID, STOP_ID)

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel('INFO')


def get_vocabulary(in_dataset, in_result_folder, in_config):
    MAX_VOCABULARY_SIZE = in_config['vocabulary_size']
    vocabulary_path = path.join(in_result_folder, 'vocab.txt')
    if not path.exists(vocabulary_path):
        logger.info('Creating vocabulary')
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
                print >> vocab_out, word
    else:
        with getreader('utf-8')(open(vocabulary_path)) as vocab_in:
            vocabulary = {}
            for line, line_index in zip(vocab_in, count()):
                vocabulary[line.strip()] = line_index
        logger.info('Skipping vocabulary creation step'.format(vocabulary_path))
    return vocabulary


def get_embedding_matrix(in_vocabulary, in_config):
    embeddings_file = in_config['embedding_matrix']
    if not path.exists(embeddings_file):
        logger.info('Creating embeddings matrix')
        w2v = Word2Vec.load_word2vec_format(
            '../word2vec_google_news/GoogleNews-vectors-negative300.bin',
            binary=True
        )
        embedding_matrix = build_embedding_matrix(w2v, in_vocabulary)
        with open(embeddings_file, 'wb') as embeddings_out:
            np.save(embeddings_out, embedding_matrix)
    else:
        logger.info('Skipping embeddings creating step')
        embedding_matrix = np.load(embeddings_file)
    return embedding_matrix


def get_bucketed_datasets(in_vocabulary, in_dataset, in_result_folder, in_config):
    logger.info('Creating the result bucketed/padded datasets')

    process_utterance = lambda seq: utterance_to_ids(seq, in_vocabulary)

    modified_dataset = in_dataset.apply(lambda row: map(process_utterance, row))
    modified_dataset[0] = modified_dataset[0].apply(lambda row: row + [GO_ID])
    modified_dataset[1] = modified_dataset[1].apply(lambda row: row + [STOP_ID])

    TESTSET_RATIO = in_config['testset_ratio']
    testset_size = int(TESTSET_RATIO * in_dataset.shape[0])
    test_set = modified_dataset.iloc[:testset_size, ]
    train_set = modified_dataset.iloc[testset_size:, ]

    train_buckets = pad_and_bucket_dataset(train_set, in_vocabulary, in_config)
    test_buckets = pad_and_bucket_dataset(test_set, in_vocabulary, in_config)
    return train_buckets, test_buckets


def pad_and_bucket_dataset(in_dataset, in_vocabulary, in_config):
    BUCKETS = in_config['buckets']
    bucket_stats = collect_bucket_stats(in_dataset, BUCKETS)
    bucketed_encoder_inputs = [
        np.zeros((bucket_stats[bucket_id], input_length), dtype=np.uint32)
        for bucket_id, (input_length, output_length) in enumerate(BUCKETS)
    ]
    bucketed_decoder_inputs = [
        np.zeros((bucket_stats[bucket_id], output_length), dtype=np.uint32)
        for bucket_id, (input_length, output_length) in enumerate(BUCKETS)
    ]
    bucket_cursors = [0 for _ in BUCKETS]
    for row in in_dataset.itertuples():
        encoder_input_ids, decoder_input_ids = row[1:]
        bucket_id = find_bucket(
            len(encoder_input_ids),
            len(decoder_input_ids),
            BUCKETS
        )
        if bucket_id is None:
            continue
        bucket_cursor = bucket_cursors[bucket_id]
        input_length, output_length = BUCKETS[bucket_id]
        padded_encoder_input = pad_sequence(
            encoder_input_ids,
            input_length,
            padding='pre'
        )
        padded_decoder_input = pad_sequence(decoder_input_ids, output_length)

        bucketed_encoder_inputs[bucket_id][bucket_cursor] = padded_encoder_input
        bucketed_decoder_inputs[bucket_id][bucket_cursor] = padded_decoder_input
        bucket_cursors[bucket_id] += 1

    return {
        'encoder': [
            np.asarray(input_bucket)
            for input_bucket in bucketed_encoder_inputs
        ],
        'decoder': [
            np.asarray(input_bucket)
            for input_bucket in bucketed_decoder_inputs
        ],
    }


def main(in_dataset, in_result_folder, in_config):
    if not path.exists(in_result_folder):
        makedirs(in_result_folder)
    vocabulary = get_vocabulary(in_dataset, in_result_folder, in_config)
    embeddings = get_embedding_matrix(in_dataset, in_config)
    train_buckets, test_buckets = get_bucketed_datasets(
        vocabulary,
        in_dataset,
        in_result_folder,
        in_config
    )
    for component_name, input_buckets in train_buckets.iteritems():
        for bucket_index, bucket in enumerate(input_buckets):
            file_path = path.join(
                in_result_folder,
                'train_{}_{}.npy'.format(component_name, bucket_index)
            )
            np.save(file_path, bucket)
    for component_name, input_buckets in test_buckets.iteritems():
        for bucket_index, bucket in enumerate(input_buckets):
            file_path = path.join(
                in_result_folder,
                'test_{}_{}.npy'.format(component_name, bucket_index)
            )
            np.save(file_path, bucket)


if __name__ == '__main__':
    if len(argv) < 3:
        print 'Usage: create_dataset.py ' + \
              '<opensubtitles csv> <config json> <result folder>'
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
