from os import path

from data_utils import generate_sequences
from batch_generator import BatchGenerator
from seq2seq_dialogue import WORKING_DIR, BUCKETS, prepare_data


def test_sequence_generator():
    MODEL_FILE = path.join(WORKING_DIR, 'model.h5')
    vocab, rev_vocab, embeddings, enc_train_ids_path, dec_train_ids_path, enc_dev_ids_path, dec_dev_ids_path = prepare_data()
    train_batch_generator = BatchGenerator(
        enc_train_ids_path,
        dec_train_ids_path,
        1,
        len(vocab), BUCKETS[1]
    )
    for _, batch in zip(xrange(1000000), generate_sequences(train_batch_generator)):
        assert batch is not None
