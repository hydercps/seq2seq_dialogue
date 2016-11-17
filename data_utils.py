"""Utilities for downloading data from WMT, tokenizing, vocabularies."""

from os import path
import re
from logging import getLogger
from collections import defaultdict

import numpy as np

from keras.preprocessing.sequence import pad_sequences

logger = getLogger()
logger.setLevel('INFO')

# Special vocabulary symbols - we always put them at the start.
PAD = '__PAD__'
GO = '__GO__'
STOP = '__STOP__'
UNK = '__UNK__'
START_VOCAB = [PAD, GO, STOP, UNK]

PAD_ID = 0
GO_ID = 1
STOP_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
DIGIT_RE = re.compile(br"\d")


def find_bucket(in_src_length, in_tgt_length, in_buckets):
    for bucket_index, (source_size, target_size) in enumerate(in_buckets):
        if in_src_length < source_size and in_tgt_length < target_size:
            return bucket_index


def pad_sequence(
    in_sequence,
    in_pad_length,
    vocabulary_size=None,
    to_onehot=False
):
    sequence_padded = pad_sequences(
        [in_sequence],
        maxlen=in_pad_length,
        padding='post',
        dtype='int32',
        value=PAD_ID
    )[0]
    if to_onehot:
        sequence_padded = ids_to_one_hots(
            sequence_padded,
            vocabulary_size
        )
    return sequence_padded


def get_special_token_vector(in_token_id, in_embedding_size):
    assert PAD_ID <= in_token_id <= UNK_ID
    dummy_vector = np.zeros((1, in_embedding_size), dtype=np.float32)
    if in_token_id != PAD_ID:
        dummy_vector[0][in_token_id] = 1.0
    return dummy_vector


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(in_data, in_max_size, normalize_digits=True):
    vocab = defaultdict(lambda: 0)
    for line in in_data:
        for token in line.split():
            word = re.sub(DIGIT_RE, b"0", token) if normalize_digits else token
            vocab[word] += 1
    vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    logger.info('Untruncated vocabulary size : {}'.format(len(vocab_list)))
    vocab_list = vocab_list[:in_max_size]
    return vocab_list


def initialize_vocabulary(vocabulary_path):
    if path.exists(vocabulary_path):
        rev_vocab = []
        with open(vocabulary_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(
    sentence,
    vocabulary,
    tokenizer=None,
    normalize_digits=True
):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(DIGIT_RE, b"0", w), UNK_ID) for w in words]


def data_to_token_ids(
    data_path,
    target_path,
    vocabulary_path,
    tokenizer=None,
    normalize_digits=True
):
    if not path.exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with open(data_path, mode="rb") as data_file, open(target_path, mode="w") as tokens_file:
            counter = 0
            for line in data_file:
                counter += 1
                if counter % 100000 == 0:
                    print("  tokenizing line %d" % counter)
                token_ids = sentence_to_token_ids(
                    line,
                    vocab,
                    tokenizer,
                    normalize_digits
                )
                tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def pad_and_bucket(source_path, target_path, buckets, max_size=None):
    """Read data from source and target files and put into buckets.

    Args:
        source_path: path to the files with token-ids for the source language.
        target_path: path to the file with token-ids for the target language;
            it must be aligned with the source file: n-th line contains the desired
            output for n-th line from the source_path.
        max_size: maximum number of lines to read, all other will be ignored;
            if 0 or None, data files will be read completely (no limit).

    Returns:
        data_set: a list of length len(_buckets); data_set[n] contains a list of
            (source, target) pairs read from the provided data files that fit
            into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
            len(target) < _buckets[n][1]; source and target are lists of token-ids.
    """
    # data_set = [[] for _ in buckets]
    data_set = {
        bucket: {
            'inputs': [],
            'outputs': []
        }
        for bucket in buckets
    }

    with open(source_path) as source_file, open(target_path) as target_file:
        source, target = source_file.readline(), target_file.readline()
        counter = 0
        while source and target and (not max_size or counter < max_size):
            counter += 1
            source_ids = [int(x) for x in source.split()[:-1]]
            target_ids = [GO_ID] + [int(x) for x in target.split()] + [STOP_ID]

            bucket_id = find_bucket_for_specific_lengths(
                len(source_ids),
                len(target_ids),
                buckets
            )
            if not bucket_id:
                raise RuntimeError('Couldn\'t find bucket for the sequence')
            source_size, target_size = buckets[bucket_id]
            source_ids_padded = pad_sequences(
                [source_ids],
                maxlen=source_size,
                padding='pre',
                dtype='int32',
                value=PAD_ID
            )[0]
            target_ids_padded = pad_sequences(
                [target_ids],
                maxlen=target_size,
                padding='post',
                dtype='int32',
                value=PAD_ID
            )[0]
            data_set[(source_size, target_size)]['inputs'].append(
                source_ids_padded
            )
            data_set[(source_size, target_size)]['outputs'].append(
                target_ids_padded
            )
            source, target = source_file.readline(), target_file.readline()
    return data_set


def collect_bucket_stats(in_dataset, in_buckets):
    result = defaultdict(lambda: 0)
    for row in in_dataset.itertuples():
        encoder_input_ids, decoder_input_ids = row[1:3]
        bucket_id = find_bucket(
            len(encoder_input_ids),
            len(decoder_input_ids),
            in_buckets
        )
        if bucket_id is None:
            logger.warn('Could not find a bucket for lengths ({}, {})'.format(
                len(encoder_input_ids),
                len(decoder_input_ids)
            ))
            continue
        result[bucket_id] += 1
    return result


def ids_to_one_hots(in_ids_list, in_vocabulary_size):
    if not len(in_ids_list):
        return None
    result = []
    for token_index, token_id in enumerate(in_ids_list):
        token_array = np.zeros((in_vocabulary_size), dtype=np.int32)
        token_array[token_id] = 1
        result.append(token_array)
    return result


def ids_to_embeddings(in_ids, in_embedding_matrix):
    if not len(in_ids):
        return None
    vocab_size, embedding_size = in_embedding_matrix.shape
    default_emb = np.zeros((1, embedding_size), dtype=np.float32)
    return [
        in_embedding_matrix[token_id] if token_id < vocab_size else default_emb
        for token_id in in_ids
    ]


def build_embedding_matrix(in_w2v_model, in_base_vocabulary=None):
    EMBEDDING_DIM = in_w2v_model.vector_size
    # if a base vocabulary is given, only building embeddings for words in it
    if in_base_vocabulary:
        result = np.zeros(
            (len(in_base_vocabulary) + 1, EMBEDDING_DIM),
            dtype=np.float32
        )
        for word_index, word in enumerate(in_base_vocabulary):
            if word in in_w2v_model:
                # words not found in embedding index will be all-zeros.
                result[word_index] = in_w2v_model[word]
        # adding dummy vectors for special tokens
        for word_id in [PAD_ID, GO_ID, STOP_ID, UNK_ID]:
            result[word_index] = get_special_token_vector(
                word_id,
                EMBEDDING_DIM
            )
    else:
        result = np.zeros(
            (len(in_w2v_model.vocab), in_w2v_model.vector_size),
            dtype=np.float32
        )
        for word_index, word in enumerate(in_w2v_model.vocab.keys()):
            result[word_index] = in_w2v_model[word]
    return result


def truncate_decoded_sequence(in_sequence):
    result = in_sequence
    stop_index = list(in_sequence).index(STOP_ID)
    if stop_index != -1:
        result = result[:stop_index]
    result = filter(lambda token_id: token_id != PAD_ID, result)
    return result

