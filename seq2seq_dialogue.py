from codecs import getreader
from os import makedirs, path
from sys import argv, stdin
import logging
from json import load

import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Activation

from seq2seq.models import AttentionSeq2Seq
from batch_generator import BatchGenerator, generate_sequences
from data_utils import truncate_decoded_sequence, find_bucket, pad_sequence, \
    GO_ID

logging.getLogger().setLevel('INFO')

# TODO: process all buckets
BUCKET_ID = 0

class Mode(object):
    TRAIN = 0
    TEST = 1


def create_model(
    in_encoder_vocabulary,
    in_decoder_vocabulary,
    in_embedding_matrix,
    in_input_length,
    in_output_length,
    mode=Mode.TRAIN
):
    effective_vocabulary_size, embedding_size = in_embedding_matrix.shape
    embedding_layer = Embedding(
        effective_vocabulary_size,
        embedding_size,
        weights=[in_embedding_matrix],
        input_length=in_input_length,
        trainable=True,
        name='emb'
    )
    seq2seq_model = AttentionSeq2Seq(
        bidirectional=False,
        input_dim=embedding_size,
        output_dim=len(in_decoder_vocabulary),
        hidden_dim=32,
        output_length=in_output_length,
        depth=1,
        dropout=0.0 if mode == Mode.TEST else 0.2
    )
    model = Sequential()
    model.add(embedding_layer)
    model.add(seq2seq_model)
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def live_decode(in_vocabulary, in_embeddings, in_config):
    logging.info('Loading the trained model')

    model = create_model(
        in_vocabulary,
        in_vocabulary,
        in_embeddings,
        in_config['buckets'][BUCKET_ID][0],
        in_config['buckets'][BUCKET_ID][1],
        mode=Mode.TEST
    )
    MODEL_FILE = in_config['model_weights']
    model.load_weights(MODEL_FILE)

    vocabulary_map = {
        token: index
        for index, token in enumerate(in_vocabulary)
    }
    print 'go'
    while True:
        user_input = stdin.readline().lower().strip()
        if user_input == 'q':
            break
        token_ids = [
            vocabulary_map[token]
            for token in user_input.split()
        ] + [GO_ID]
        BUCKETS = in_config['buckets']
        bucket_id = find_bucket(len(token_ids), 0, BUCKETS)
        decoder_inputs = pad_sequence(
            token_ids,
            BUCKETS[bucket_id][0],
            padding='pre'
        )
        decoder_input_matrix = np.asarray(decoder_inputs)
        decoder_input_matrix = decoder_input_matrix.reshape(
            [1] + list(decoder_input_matrix.shape)
        )
        decoder_outputs = model.predict(decoder_input_matrix)
        decoded_ids = truncate_decoded_sequence(
            np.argmax(decoder_outputs, axis=1)[0]
        )
        print ' '.join([
            in_vocabulary[decoded_token]
            for decoded_token in decoded_ids
        ])


def train(in_vocabulary, in_embeddings, in_config):
    logging.info('Training the model')
    model = create_model(
        in_vocabulary,
        in_vocabulary,
        in_embeddings,
        in_config['buckets'][BUCKET_ID][0],
        in_config['buckets'][BUCKET_ID][1]
    )
    encoder_input_file = path.join(
        in_config['data_folder'],
        'train_encoder_{}.npy'.format(BUCKET_ID)
    )
    decoder_input_file = path.join(
        in_config['data_folder'],
        'train_decoder_{}.npy'.format(BUCKET_ID)
    )
    train_batch_generator = BatchGenerator(
        encoder_input_file,
        decoder_input_file,
        in_config['batch_size'],
        in_vocabulary
    )
    MODEL_FILE = in_config['model_weights']
    save_callback = ModelCheckpoint(
        MODEL_FILE,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto'
    )
    model.fit_generator(
        generate_sequences(train_batch_generator),
        nb_epoch=in_config['nb_epoch'],
        samples_per_epoch=in_config['samples_per_epoch'],
        callbacks=[save_callback]
    )
    evaluate(in_vocabulary, in_embeddings, in_config)


def evaluate(in_vocabulary, in_embeddings, in_config):
    logging.info('Evaluating the trained model')

    model = create_model(
        in_vocabulary,
        in_vocabulary,
        in_embeddings,
        in_config['buckets'][BUCKET_ID][0],
        in_config['buckets'][BUCKET_ID][1],
        mode=Mode.TEST
    )
    MODEL_FILE = in_config['model_weights']
    model.load_weights(MODEL_FILE)

    encoder_input_file = path.join(
        in_config['data_folder'],
        'test_encoder_{}.npy'.format(BUCKET_ID)
    )
    decoder_input_file = path.join(
        in_config['data_folder'],
        'test_decoder_{}.npy'.format(BUCKET_ID)
    )
    test_batch_generator = BatchGenerator(
        encoder_input_file,
        decoder_input_file,
        in_config['batch_size'],
        in_vocabulary
    )
    print model.evaluate_generator(
        generate_sequences(test_batch_generator),
        test_batch_generator.get_dataset_size()
    )


def main(in_mode, in_config):
    MODEL_FILE = in_config['model_weights']
    MODEL_DIR = path.dirname(MODEL_FILE)
    if not path.exists(MODEL_DIR):
        makedirs(MODEL_DIR)
    with getreader('utf-8')(open(in_config['vocabulary'])) as vocab_in:
        VOCAB = [line.strip() for line in vocab_in]
    EMBEDDINGS = np.load(in_config['embedding_matrix'])
    if in_mode == 'train':
        train(VOCAB, EMBEDDINGS, in_config)
    if in_mode == 'test':
        evaluate(VOCAB, EMBEDDINGS, in_config)
    if in_mode == 'live_decode':
        live_decode(VOCAB, EMBEDDINGS, in_config)


if __name__ == '__main__':
    if len(argv) < 3:
        print 'Usage: seq2seq_dialogue.py <train/test/live_decode> <config file>'
        exit()
    mode, config_file = argv[1:3]
    with getreader('utf-8')(open(config_file)) as config_in:
        config = load(config_in)
    main(mode, config)
