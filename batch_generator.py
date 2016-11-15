import numpy as np


# Generating a batch of sequences for the specific bucket
class BatchGenerator(object):
    def __init__(
        self,
        in_encoder_input,
        in_decoder_input,
        in_batch_size,
        in_vocabulary,
        in_bucket,
        in_embeddings=None
    ):
        self.encoder_input = np.load(in_encoder_input, mmap_mode='r')
        self.decoder_input = np.load(in_decoder_input, mmap_mode='r')
        self.batch_size = in_batch_size
        self.reverse_vocabulary = {
            word: index for index, word in enumerate(in_vocabulary)
        }
        self.embeddings = in_embeddings
        self.bucket = in_bucket
        self.cursor = 0

    def generate_batch(self):
        if self.encoder_input.shape[0] < self.cursor + self.batch_size:
            self.cursor = 0
        X = self.encoder_input[self.cursor: self.cursor + self.batch_size]
        y = self.decoder_input[self.cursor: self.cursor + self.batch_size]
        return X, y

    def get_dataset_size(self):
        return self.encoder_input.shape[0]


def generate_sequences(in_batch_generator):
    while True:
        yield in_batch_generator.generate_batch()
