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
    word1 = tf.keras.layers.Embedding(input_dim=params.vocab_size+1, output_dim=params.dimension_size, input_length=params.max_subword)
    word2 = tf.keras.layers.Embedding(input_dim=params.vocab_size+1, output_dim=params.dimension_size, input_length=params.max_subword)

    # Input Shape: 32, 15, 100 -> batch_size, subwords_count, vector dimension
    logging.info('word1 shape: {}'.format(word1.shape))
    logging.info('word2 shape: {}'.format(word2.shape))

    # Example - From (64, 10, 300) TO (64, 300, 10)
    # temp = keras.layers.Permute((2, 1))(inputs)
    # logging.info('temp: ', temp.shape)

    nConv = params.numConv
    filer = params.filter
    kernel = params.kernel
    temp = inputs

    for i in range(nConv):

        temp = keras.layers.Conv2D(
            filters=filer * (2**i), kernel_size=kernel, activation='relu')(temp)
        logging.info('After Conv {} shape: {}'.format(i, temp.shape))

        temp = keras.layers.MaxPooling2D(
            pool_size=(2,2))(temp)
        logging.info('After Max Pool {} shape: {} '.format(i, temp.shape))

        # temp = keras.layers.BatchNormalization()(temp)

    # 1x1 convolution network given patch sizes
    yPatches = keras.layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid')(temp)
    
    flat = keras.layers.Flatten()(yPatches)

    Y = SWlayer.SWAP(output_dim=1, bias=True)(flat)
    print('Patches Y: {}'.format(Y.shape))
    
    probabiltyStatus = keras.layers.Activation('sigmoid')(Y)

    logging.info('################### Heat Map Model #################')
    modelHeatMap = Model(inputs=inputs, outputs=[yPatches])

    logging.info('################### Label Prediction Model #################')
    modelLabel = Model(inputs=inputs, outputs=[probabiltyStatus])

    return modelLabel, modelHeatMap
