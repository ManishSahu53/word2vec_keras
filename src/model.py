import logging
import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf


import config
from src import util
from src import preprocessing
from src import SWlayer


# Retina Eye Model
def word2vecModel(params):
    input_word1 = tf.keras.Input(shape=(1,))
    input_word2 = tf.keras.Input(shape=(1,))

    embedding_word1 = tf.keras.layers.Embedding(input_dim=params.vocab_size+1, output_dim=params.dimension_size, input_length=params.max_subword)
    embedding_word2 = tf.keras.layers.Embedding(input_dim=params.vocab_size+1, output_dim=params.dimension_size, input_length=params.max_subword)

    word1 = embedding_word1(input_word1)
    word2 = embedding_word2(input_word2)

    # Input Shape: 32, 15, 100 -> batch_size, subwords_count, vector dimension
    logging.info('word1 shape: {}'.format(word1.shape))
    logging.info('word2 shape: {}'.format(word2.shape))

    dot = tf.keras.layers.Dot(axes=1, normalize=True)

    cosine_distance = dot([word1, word2])

    model = tf.keras.Model(inputs=[input_word1, input_word2], outputs=[cosine_distance])

    return model
    
    # Example - From (64, 10, 300) TO (64, 300, 10)
    # temp = keras.layers.Permute((2, 1))(inputs)
    # logging.info('temp: ', temp.shape)
    

