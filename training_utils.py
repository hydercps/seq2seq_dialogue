import numpy as np

from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers.core import Activation
from keras.optimizers import SGD

from seq2seq.models import AttentionSeq2Seq
from data_utils import truncate_decoded_sequence


class Mode(object):
    TRAIN = 0
    TEST = 1


def create_model(
    in_encoder_vocabulary,
    in_decoder_vocabulary,
    in_embedding_matrix,
    in_input_length,
    in_output_length,
    in_config,
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
        hidden_dim=in_config['layer_size'],
        output_length=in_output_length,
        depth=in_config['max_layers'],
        dropout=0.0 if mode == Mode.TEST else 0.2
    )
    model = Sequential()
    model.add(embedding_layer)
    model.add(seq2seq_model)
    model.add(Activation('softmax'))
    sgd = SGD(
        lr=in_config['learning_rate'],
        decay=in_config['learning_rate_decay'],
        clipvalue=in_config['gradient_clip_value']
    )
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model


class DecodingDemo(Callback):
    def __init__(self, in_vocabulary, in_embeddings, in_bucket_id, in_config, in_demo_inputs):
        self.vocabulary = in_vocabulary
        self.demo_inputs = in_demo_inputs
        input_length, output_length = in_config['buckets'][in_bucket_id]
        self.demo_model = create_model(
            self.vocabulary,
            self.vocabulary,
            in_embeddings,
            input_length,
            output_length,
            in_config,
            mode=Mode.TEST
        )

    def on_batch_end(self, batch, logs):
        self.demo_model.set_weights(self.model.get_weights()) 
        print 'Decoding demo:' 
        for input in self.demo_inputs:
            output_softmaxes = self.demo_model.predict(input.reshape([1] + list(input.shape)))
            truncated_output = truncate_decoded_sequence(
                np.argmax(output_softmaxes[0], axis=1)
            )
            print ' '.join([self.vocabulary[token_id] for token_id in truncated_output]) 

