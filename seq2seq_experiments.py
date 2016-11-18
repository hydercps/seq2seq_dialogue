
# coding: utf-8

# In[27]:

import numpy as np
from codecs import getreader, getwriter
from os import path, makedirs

from data_utils import PAD_ID


# In[28]:

encoder_dataset, decoder_dataset = np.load('dataset/train_encoder_0.npy'), np.load('dataset/train_decoder_0.npy')
with getreader('utf-8')(open('dataset/vocab.txt')) as vocab_in:
    vocabulary = [line.strip() for line in vocab_in]


# In[29]:

SANITY_SET_SIZE = 200
RESULT_FOLDER = 'sanity_check'

if not path.exists(RESULT_FOLDER):
    makedirs(RESULT_FOLDER)

encoder_data, decoder_data = encoder_dataset[:SANITY_SET_SIZE], decoder_dataset[:SANITY_SET_SIZE]
with getwriter('utf-8')(open(path.join(RESULT_FOLDER, 'train_0.txt'), 'w')) as dataset_out:
    for encoder_sequence, decoder_sequence in zip(encoder_data, decoder_data):
        encoder_string = ' '.join([
            vocabulary[token_id]
            for token_id in encoder_sequence
            # if token_id not in [PAD_ID]
        ])
        decoder_string = ' '.join([
            vocabulary[token_id]
            for token_id in decoder_sequence
            # if token_id not in [PAD_ID]
        ])
        print >>dataset_out, u';'.join([encoder_string, decoder_string])

np.save(path.join(RESULT_FOLDER, 'train_encoder_0.txt'), encoder_data)
np.save(path.join(RESULT_FOLDER, 'train_decoder_0.txt'), decoder_data)

