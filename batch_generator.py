import numpy as np

from keras.preprocessing.sequence import pad_sequences
from data_utils import PAD_ID, ids_to_embeddings, ids_to_one_hots


# Generating a batch of sequences for the specific bucket
class BatchGenerator(object):
    def __init__(
        self,
        in_enc_filename,
        in_dec_filename,
        in_batch_size,
        in_vocabulary_size,
        in_bucket,
        in_embeddings=None
    ):
        self.encoder_src = open(in_enc_filename)
        self.decoder_src = open(in_dec_filename)
        self.batch_size = in_batch_size
        self.vocabulary_size = in_vocabulary_size
        self.embeddings = in_embeddings
        self.bucket = in_bucket

    def generate_batch(self):
        x_list, y_list = [], []
        while len(x_list) < self.batch_size:
            encoder_line = self.encoder_src.readline()
            decoder_line = self.decoder_src.readline()
            if not encoder_line or not decoder_line:
                self.__reload_sources()
                continue
            encoder_input = map(int, encoder_line.split())
            decoder_input = map(int, decoder_line.split())
            if (
                len(encoder_input) < self.bucket[0] and
                len(decoder_input) < self.bucket[1]
            ):
                x_list.append(encoder_input)
                y_list.append(decoder_input)
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

    def sequence_to_vector(
        self,
        in_sequence,
        in_max_sequence_length,
        to_onehot=False
    ):
        sequence_padded = pad_sequences(
            [in_sequence],
            maxlen=in_max_sequence_length,
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
                self.vocabulary_size
            )
        return np.asarray(sequence_padded)

    def __reload_sources(self):
        self.encoder_src.seek(0, 0)
        self.decoder_src.seek(0, 0)


def generate_sequences(in_batch_generator):
    while True:
        yield in_batch_generator.generate_batch()