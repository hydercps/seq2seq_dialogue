from os import getcwd, path, makedirs
from sys import argv
import logging

import numpy as np

from keras.models import Sequential
from keras.layers import Embedding

from gensim.models import Word2Vec
from seq2seq.models import AttentionSeq2Seq

from data_utils import prepare_custom_data, START_VOCAB
from batch_generator import BatchGenerator, generate_sequences

logging.getLogger().setLevel('INFO')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]

VOCABULARY_SIZE = 40000 + len(START_VOCAB)
LAYER_SIZE = 256
MAX_LAYERS = 3
MAX_GRADIENT_NORM = 5.0
BATCH_SIZE = 8
LEARNING_RATE = 9.0
LEARNING_RATE_DECAY_FACTOR = 0.01
FORWARD_ONLY = False

DATASET_DIR = path.join(
    getcwd(),
    '..',
    'opensubtitles_tools',
    'opensubtitles_seq2seq_dataset'
)
WORKING_DIR = path.join(getcwd(), 'result')
WORD2VEC_MODEL_PATH = path.join(
    getcwd(),
    '..',
    'word2vec_google_news',
    'GoogleNews-vectors-negative300.bin'
)


def create_model(
    in_encoder_vocabulary,
    in_decoder_vocabulary,
    in_embedding_matrix,
    mode='train'
):
    effective_vocabulary_size, embedding_size = in_embedding_matrix.shape
    embedding_layer = Embedding(
        effective_vocabulary_size,
        embedding_size,
        weights=[in_embedding_matrix],
        input_length=BUCKETS[1][0],
        trainable=True,
        name='emb'
    )
    seq2seq_model = AttentionSeq2Seq(
        bidirectional=False,
        input_dim=embedding_size,
        output_dim=len(in_decoder_vocabulary),
        hidden_dim=32,
        output_length=BUCKETS[1][1],
        depth=1,
        dropout=0.0 if mode == 'test' else 0.2
    )
    model = Sequential()
    model.add(embedding_layer)
    model.add(seq2seq_model)
    model.compile(loss='mse', optimizer='sgd')
    return model


def prepare_data():
    # prepare dataset
    logging.info('Preparing data')
    if not path.exists(WORKING_DIR):
        makedirs(WORKING_DIR)
    return prepare_custom_data(
        WORKING_DIR,
        path.join(DATASET_DIR, 'train.enc'),
        path.join(DATASET_DIR, 'train.dec'),
        path.join(DATASET_DIR, 'test.enc'),
        path.join(DATASET_DIR, 'test.dec'),
        WORD2VEC_MODEL_PATH,
        path.join(DATASET_DIR, 'embeddings.npy')
    )


def train(train_set, dev_set):
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(BUCKETS))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [
        sum(train_bucket_sizes[:i + 1]) / train_total_size
        for i in xrange(len(train_bucket_sizes))
    ]

    model = create_model()
    # Choose a bucket according to data distribution. We pick a random number
    # in [0, 1] and use the corresponding interval in train_buckets_scale.
    random_number_01 = np.random.random_sample()
    bucket_id = min([
        i for i in xrange(len(train_buckets_scale))
        if random_number_01 < train_buckets_scale[i]
    ])
    # Get a batch and make a step
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(
        train_set,
        bucket_id
    )
    model.fit(encoder_inputs, decoder_inputs)
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


def main(in_command):
    MODEL_FILE = path.join(WORKING_DIR, 'model.h5')
    (
        vocab,
        rev_vocab,
        embeddings,
        enc_train_ids_path,
        dec_train_ids_path,
        enc_dev_ids_path,
        dec_dev_ids_path
    ) = prepare_data()
    if in_command == 'train':
        model = create_model(vocab, vocab, embeddings)
        train_batch_generator = BatchGenerator(
            enc_train_ids_path,
            dec_train_ids_path,
            BATCH_SIZE,
            len(vocab), BUCKETS[1]
        )
        # import pdb; pdb.set_trace()
        # X, y = train_batch_generator.generate_batch()
        import pdb; pdb.set_trace()
        model.fit_generator(
            generate_sequences(train_batch_generator),
            nb_epoch=2,
            samples_per_epoch=32
        )
        # model.train_on_batch(X, y)
        # model.fit_generator(generate_sequences(train_batch_generator), 100, 2)
        model.save_weights(MODEL_FILE, overwrite=True)
        del model
        model = create_model(vocab, vocab, embeddings, mode='test')
        model.load_weights(MODEL_FILE)
        print model.evaluate(
            dev_set[BUCKETS[1]]['inputs'],
            dev_set[BUCKETS[1]]['outputs'],
            batch_size=16,
            verbose=True
        )
    if in_command == 'test':
        model = create_model(vocab, vocab, embeddings, mode='test')
        model.load_weights(MODEL_FILE)
        w2v = Word2Vec.load_word2vec_format(WORD2VEC_MODEL_PATH, binary=True)
        predictions = model.predict(dev_set[BUCKETS[1]]['inputs'])
        for vector_index in xrange(predictions.shape[0]):
            print visualize_decoded(vocab, w2v, predictions[vector_index])


if __name__ == '__main__':
    if len(argv) < 2:
        print 'Usage: seq2seq_dialogue.py <train|test>'
        exit()
    command = argv[1].lower()
    main(command)

