from codecs import getreader
from os import makedirs, path
from sys import argv
import logging
from json import load

import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Activation

from seq2seq.models import AttentionSeq2Seq
from batch_generator import BatchGenerator, generate_sequences

logging.getLogger().setLevel('INFO')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.


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


def decode():
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    enc_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.enc" % gConfig['enc_vocab_size'])
    dec_vocab_path = os.path.join(gConfig['working_directory'],"vocab%d.dec" % gConfig['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

    # Decode from standard input.
    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
      # Get token-ids for the input sentence.
      token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
      # Which bucket does it belong to?
      bucket_id = min([b for b in xrange(len(_buckets))
                       if _buckets[b][0] > len(token_ids)])
      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      # If there is an EOS symbol in outputs, cut them at that point.
      if data_utils.EOS_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])
      print "> "
      sys.stdout.flush()
      sentence = sys.stdin.readline()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.initialize_all_variables())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in xrange(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def decode_line(model, enc_vocab, in_embeddings, sentence):
    words = sentence.split()
    decoder_inputs = np.zeros(
        (1, len(words), in_embeddings.shape[1]),
        dtype=np.int32
    )
    for word_index, word in enumerate(words):
        token_id = enc_vocab[word.lower()]
        decoder_inputs[0][word_index][token_id] = 1
    decoder_outputs = model.predict(decoder_inputs)
    result = ' '.join([
        in_embeddings.similar_by_vector(output)[0][0]
        for output in decoder_outputs
    ])
    return result


def visualize_decoded(in_vocab, in_w2v, in_decoder_outputs):
    result = ' '.join([
        in_w2v.similar_by_vector(in_decoder_outputs[vector_index])[0][0]
        for vector_index in xrange(in_decoder_outputs.shape[0])
    ])
    return result


def train(in_vocabulary, in_embeddings, in_config):
    logging.info('Training the model')
    model = create_model(
        in_vocabulary,
        in_vocabulary,
        in_embeddings,
        in_config['buckets'][1][0],
        in_config['buckets'][1][1]
    )
    bucket = 1
    encoder_input_file = path.join(
        in_config['data_folder'],
        'train_encoder_{}.npy'.format(bucket)
    )
    decoder_input_file = path.join(
        in_config['data_folder'],
        'train_decoder_{}.npy'.format(bucket)
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
    BUCKET = 1
    model = create_model(
        in_vocabulary,
        in_vocabulary,
        in_embeddings,
        in_config['buckets'][BUCKET][0],
        in_config['buckets'][BUCKET][1],
        mode=Mode.TEST
    )
    MODEL_FILE = in_config['model_weights']
    model.load_weights(MODEL_FILE)

    encoder_input_file = path.join(
        in_config['data_folder'],
        'test_encoder_{}.npy'.format(BUCKET)
    )
    decoder_input_file = path.join(
        in_config['data_folder'],
        'test_decoder_{}.npy'.format(BUCKET)
    )
    test_batch_generator = BatchGenerator(
        encoder_input_file,
        decoder_input_file,
        in_config['batch_size'],
        in_vocabulary
    )
    print model.evaluate_generator(
        test_batch_generator,
        test_batch_generator.get_dataset_size()
    )


def main(in_mode, in_config):
    MODEL_FILE = in_config['model_weights']
    MODEL_DIR = path.dirname(MODEL_FILE)
    if not path.exists(MODEL_DIR):
        makedirs(MODEL_DIR)
    with getreader('utf-8')(open(in_config['vocabulary'])) as vocab_in:
        VOCAB = [line.strip() for line in vocab_in]
    EMBEDDINGS = np.load(in_config['embeddings_matrix'])
    if in_mode == 'train':
        train(VOCAB, EMBEDDINGS, in_config)
    if in_mode == 'test':
        evaluate(VOCAB, EMBEDDINGS, in_config)


if __name__ == '__main__':
    if len(argv) < 3:
        print 'Usage: seq2seq_dialogue.py <train/test> <config file>'
        exit()
    mode, config_file = argv[1:3]
    with getreader('utf-8')(open(config_file)) as config_in:
        config = load(config_in)
    main(mode, config)
