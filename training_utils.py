import numpy as np
from keras.callbacks import Callback

from data_utils import truncate_decoded_sequence


class DecodingDemo(Callback):
    def __init__(self, in_vocabulary, in_demo_inputs):
        self.vocabulary = in_vocabulary
        self.demo_inputs = in_demo_inputs

    def on_batch_end(self, batch):
        print 'Decoding demo:' 
        for input in in_demo_inputs;
            output_softmaxes = self.model.predict(input.reshape([1] + input.shape))
            truncated_output = truncate_decoded_sequence(
                np.argmax(output_softmaxes[0], axis=1)
            )
            print ' '.join([self.vocabulary[token_id] for token_id in truncated_output]) 

