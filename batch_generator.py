import numpy as np
from keras.preprocessing.sequence import pad_sequences
from pandas import read_csv

from data_utils import PAD_ID, ids_to_embeddings, ids_to_one_hots


# Generating a batch of sequences for the specific bucket
class BatchGenerator(object):
    def __init__(
        self,
        in_dataset_file,
        in_batch_size,
        in_vocabulary,
        in_bucket,
        in_embeddings=None
    ):
        self.dataset = read_csv(
            in_dataset_file,
            header=None,
            sep=';',
            encoding='utf-8',
            dtype=str
        )
        self.dataset.fillna('')
        self.batch_size = in_batch_size
        self.reverse_vocabulary = {
            word: index for index, word in enumerate(in_vocabulary)
        }
        self.embeddings = in_embeddings
        self.bucket = in_bucket
        self.cursor = 0

    def generate_batch(self):
        x_list, y_list = [], []
        while len(x_list) < self.batch_size:
            encoder_line, decoder_line = self.dataset.ix[self.cursor,]
            encoder_input = map(
                self.reverse_vocabulary.get,
                encoder_line.split()
            )
            decoder_input = map(
                self.reverse_vocabulary.get,
                decoder_line.split()
            )
            if (
                len(encoder_input) < self.bucket[0] and
                len(decoder_input) < self.bucket[1]
            ):
                x_list.append(encoder_input)
                y_list.append(decoder_input)
            self.cursor = (self.cursor + 1) % self.dataset.shape[0]
        # it's a full batch or nothing at all
        if len(x_list) < self.batch_size:
            raise RuntimeError('File contents insufficient for a batch')
        X = [
            self.sequence_to_vector(sequence, self.bucket[0])
            for sequence in x_list
        ]
        y = [
            self.sequence_to_vector(sequence, self.bucket[1], to_onehot=True)
            for sequence in y_list
        ]
        return np.asarray(X), np.asarray(y)

    def sequence_to_vector(self, in_sequence, in_pad_length, to_onehot=False):
        sequence_padded = pad_sequences(
            [in_sequence],
            maxlen=in_pad_length,
            padding='post',
            dtype='int32',
            value=PAD_ID
        )[0]
        if self.embeddings:
            sequence_padded = ids_to_embeddings(
                sequence_padded,
                self.embeddings
            )
        elif to_onehot:
            sequence_padded = ids_to_one_hots(
                sequence_padded,
                len(self.reverse_vocabulary)
            )
        return np.asarray(sequence_padded)


def generate_sequences(in_batch_generator):
    while True:
        yield in_batch_generator.generate_batch()