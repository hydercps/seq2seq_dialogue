from os import getcwd, path, makedirs
from sys import argv
import logging

import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Embedding

from seq2seq.models import AttentionSeq2Seq

from data_utils import pad_and_bucket, prepare_custom_data, START_VOCAB

logging.getLogger().setLevel('INFO')

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
BUCKETS = [(5, 10), (10, 15), (20, 25), (40, 50)]

VOCABULARY_SIZE = 40000 + len(START_VOCAB)
LAYER_SIZE = 256
MAX_LAYERS = 3
MAX_GRADIENT_NORM = 5.0
BATCH_SIZE = 1
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


def create_model(in_encoder_vocabulary, in_decoder_vocabulary, in_embedding_matrix):
    effective_vocabulary_size, embedding_size = in_embedding_matrix.shape
    embedding_layer = Embedding(
        len(in_encoder_vocabulary) + 1,
        embedding_size,
        weights=[in_embedding_matrix],
        input_length=BUCKETS[1][0],
        trainable=True,
        name='emb',
    )
    seq2seq_model = AttentionSeq2Seq(
        bidirectional=False,
        input_dim=embedding_size,
        input_length=BUCKETS[1][0],
        output_dim=len(in_decoder_vocabulary) + 1,
        hidden_dim=32,
        output_length=BUCKETS[1][1],
        depth=1,
        dropout=0.0
    )
    model = Sequential()
    model.add(embedding_layer)
    model.add(seq2seq_model)
    model.compile(loss='mse', optimizer='adam')
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
        path.join(DATASET_DIR, 'embeddings.npy'),
        BUCKETS
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
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(train_set, bucket_id)
    model.fit(encoder_inputs, decoder_inputs)
    return model

    with tf.Session(config=config) as sess:


        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        while True:

          start_time = time.time()
          _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, False)
          step_time += (time.time() - start_time) / gConfig['steps_per_checkpoint']
          loss += step_loss / gConfig['steps_per_checkpoint']
          current_step += 1

          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % gConfig['steps_per_checkpoint'] == 0:
            # Print statistics for the previous epoch.
            perplexity = math.exp(loss) if loss < 300 else float('inf')
            print ("global step %d learning rate %.4f step-time %.2f perplexity "
                   "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                             step_time, perplexity))
            # Decrease learning rate if no improvement was seen over last 3 times.
            if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
              sess.run(model.learning_rate_decay_op)
            previous_losses.append(loss)
            # Save checkpoint and zero timer and loss.
            checkpoint_path = path.join(gConfig['working_directory'], "seq2seq.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            step_time, loss = 0.0, 0.0
            # Run evals on development set and print their perplexity.
            for bucket_id in xrange(len(_buckets)):
              if len(dev_set[bucket_id]) == 0:
                print("  eval: empty bucket %d" % (bucket_id))
                continue
              encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                  dev_set, bucket_id)
              _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                           target_weights, bucket_id, True)
              eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
              print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            sys.stdout.flush()


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


def decode_line(sess, model, enc_vocab, rev_dec_vocab, sentence):
    # Get token-ids for the input sentence.
    token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)

    # Which bucket does it belong to?
    bucket_id = min([b for b in xrange(len(_buckets)) if len(token_ids) < BUCKETS[b][0]])

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch({bucket_id: [(token_ids, [])]}, bucket_id)

    # Get output logits for the sentence.
    _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

    # This is a greedy decoder - outputs are just argmaxes of output_logits.
    outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]

    # If there is an EOS symbol in outputs, cut them at that point.
    if data_utils.STOP_ID in outputs:
        outputs = outputs[:outputs.index(data_utils.STOP_ID)]

    return " ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs])


def main(in_command):
    MODEL_FILE = path.join(WORKING_DIR, 'model.h5')
    enc_vocab, enc_rev_vocab, dec_vocab, dec_rev_vocab, embeddings, train_set, dev_set = prepare_data()
    if in_command == 'train':
        model = create_model(enc_vocab, dec_vocab, embeddings)
        model.fit(
            train_set[BUCKETS[1]]['inputs'],
            train_set[BUCKETS[1]]['outputs'],
            batch_size=16,
            nb_epoch=1
        )
        model.save(MODEL_FILE, overwrite=True)

        print model.evaluate(
            dev_set[BUCKETS[1]]['inputs'],
            dev_set[BUCKETS[1]]['outputs'],
            batch_size=16,
            verbose=True
        )
    if in_command == 'test':
        model = load_model(MODEL_FILE)
        model.predict(dev_set[BUCKETS[1]]['inputs'][0])


if __name__ == '__main__':
    if len(argv) < 2:
        print 'Usage: seq2seq_dialogue.py <train|test>'
        exit()
    command = argv[1].lower()
    main(command)
