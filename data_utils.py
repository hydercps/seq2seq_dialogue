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
        if in_src_length + 1 < source_size and in_tgt_length < target_size:
            return bucket_index


def pad_sequence(
    in_sequence,
    in_pad_length,
    padding='post',
    vocabulary_size=None,
    to_onehot=False,
):
    sequence_padded = pad_sequences(
        [in_sequence],
        maxlen=in_pad_length,
        dtype='int32',
        value=PAD_ID,
        padding=padding
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
    if STOP_ID in in_sequence:
        stop_index = list(in_sequence).index(STOP_ID)
        result = result[:stop_index]
    result = filter(lambda token_id: token_id != PAD_ID, result)
    return result


def utterance_to_ids(in_utterance, in_vocabulary):
    return [
        in_vocabulary[token] if token in in_vocabulary else UNK_ID
        for token in in_utterance.split()
    ]
