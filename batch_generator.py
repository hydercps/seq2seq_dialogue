import numpy as np
from os import path

# Generating a batch of sequences for the specific bucket


class BatchGenerator(object):
    def __init__(
        self,
        in_encoder_input_file,
        in_decoder_input_file,
        in_batch_size,
        in_vocabulary
    ):
        self.encoder_input = np.load(in_encoder_input_file, mmap_mode='r')
        self.decoder_input = np.load(in_decoder_input_file, mmap_mode='r')
        self.batch_size = in_batch_size
        self.vocabulary = in_vocabulary
        self.cursor = 0

    def generate_batch(self):
        if self.encoder_input.shape[0] < self.cursor + self.batch_size:
            self.cursor = 0
        X = self.encoder_input[self.cursor: self.cursor + self.batch_size]
        y = np.zeros(
            (self.batch_size, self.decoder_input.shape[1], len(self.vocabulary)),
            dtype=np.uint32
        )
        for i in xrange(self.batch_size):
            for j in xrange(self.decoder_input.shape[1]):
                token_id = self.decoder_input[self.cursor + i][j]
                y[i][j][token_id] = 1
        # y = self.decoder_input[self.cursor: self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return X, y

    def get_dataset_size(self):
        return self.encoder_input.shape[0]


def generate_sequences(in_batch_generator):
    while True:
        yield in_batch_generator.generate_batch()
