"""Utilities for downloading data from WMT, tokenizing, vocabularies."""

from os import path
import re
import logging

import numpy as np

from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

logger = logging.getLogger()

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


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(
    vocabulary_path,
    data_path,
    max_vocabulary_size,
    tokenizer=None,
    normalize_digits=True
):
    if not path.exists(vocabulary_path):
        print("Creating vocabulary %s from %s" % (vocabulary_path, data_path))
        vocab = {}
        with open(data_path, mode="rb") as f:
            counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print("  processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                for w in tokens:
                    word = re.sub(DIGIT_RE, b"0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
            print('>> Full Vocabulary Size :',len(vocab_list))
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with open(vocabulary_path, mode="wb") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + b"\n")


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


def sentence_to_token_ids(sentence, vocabulary, tokenizer=None, normalize_digits=True):
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


def prepare_custom_data(
    working_directory,
    train_enc,
    train_dec,
    test_enc,
    test_dec,
    w2v_model_path,
    embeddings_path,
    in_buckets,
    tokenizer=None
):
    max_vocabulary_size = 100000
    # Create vocabularies of the appropriate sizes.
    enc_vocab_path = path.join(working_directory, "vocab.enc")
    dec_vocab_path = path.join(working_directory, "vocab.dec")
    create_vocabulary(enc_vocab_path, train_enc, max_vocabulary_size, tokenizer)
    create_vocabulary(dec_vocab_path, train_dec, max_vocabulary_size, tokenizer)

    # only building embeddings for the encoder vocabulary
    enc_vocab, enc_rev_vocab = initialize_vocabulary(enc_vocab_path)
    dec_vocab, dec_rev_vocab = initialize_vocabulary(dec_vocab_path)
    
    if not path.exists(embeddings_path):
        logger.info('Building embeddings matrix for encoder vocabulary')
        embeddings = build_embedding_matrix(
            enc_vocab,
            Word2Vec.load_word2vec_format(w2v_model_path, binary=True)
        )
        with open(embeddings_path, 'wb') as embeddings_out:
            np.save(embeddings_out, embeddings)
    else:
        embeddings = np.load(embeddings_path)

    # Create token ids for the training data.
    enc_train_ids_path = train_enc + (".ids")
    dec_train_ids_path = train_dec + (".ids")
    data_to_token_ids(train_enc, enc_train_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(train_dec, dec_train_ids_path, dec_vocab_path, tokenizer)

    # Create token ids for the development data.
    enc_dev_ids_path = test_enc + (".ids")
    dec_dev_ids_path = test_dec + (".ids")
    data_to_token_ids(test_enc, enc_dev_ids_path, enc_vocab_path, tokenizer)
    data_to_token_ids(test_dec, dec_dev_ids_path, dec_vocab_path, tokenizer)

    train_set = pad_and_bucket(enc_train_ids_path, dec_train_ids_path, in_buckets)
    dev_set = pad_and_bucket(enc_dev_ids_path, dec_dev_ids_path, in_buckets)
    for bucket in train_set:
        train_set[bucket]['inputs'] = np.asarray(train_set[bucket]['inputs'], dtype=np.int32)
        train_set[bucket]['outputs'] = ids_to_embeddings(train_set[bucket]['outputs'], embeddings)
        dev_set[bucket]['inputs'] = np.asarray(dev_set[bucket]['inputs'], dtype=np.int32)
        dev_set[bucket]['outputs'] = ids_to_embeddings(dev_set[bucket]['outputs'], embeddings)
    return enc_vocab, enc_rev_vocab, dec_vocab, dec_rev_vocab, embeddings, train_set, dev_set


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

            for source_size, target_size in data_set:
                if len(source_ids) < source_size and len(target_ids) < target_size:
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
                    break
            source, target = source_file.readline(), target_file.readline()
    return data_set


def ids_to_one_hots(in_ids_bucket, in_vocabulary_size):
    if not len(in_ids_bucket):
        return None
    result = np.zeros(
        (len(in_ids_bucket), len(in_ids_bucket[0]), in_vocabulary_size + 1), 
        dtype=np.int32
    )
    for sequence_id, sequence in enumerate(in_ids_bucket):
        for id_index, token_id in enumerate(sequence):
            result[sequence_id][id_index][token_id] = 1
    return result


def ids_to_embeddings(in_ids_bucket, in_embedding_matrix):
    if not len(in_ids_bucket):
        return None
    embedding_size = in_embedding_matrix.shape[1]
    result = np.zeros(
        (len(in_ids_bucket), len(in_ids_bucket[0]), embedding_size),
        dtype=np.float32
    )
    for sequence_id, sequence in enumerate(in_ids_bucket):
        for id_index, token_id in enumerate(sequence):
            result[sequence_id][id_index] = in_embedding_matrix[token_id] 
    return result 


def build_embedding_matrix(in_vocabulary, in_w2v_model):
    EMBEDDING_DIM = in_w2v_model.vector_size

    result = np.zeros((len(in_vocabulary) + 1, EMBEDDING_DIM), dtype=np.float32)
    for word, word_index in in_vocabulary.iteritems():
        if word in in_w2v_model:
            # words not found in embedding index will be all-zeros.
            result[word_index] = in_w2v_model[word]
    return result
