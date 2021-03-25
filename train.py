import os
import argparse
import logging

import numpy as np
import tensorflow as tf

import config

from src import model
from src import dataGenerator
from src import util
from src import model

from src import pre_processing

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# _config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Checking for logging folder and creating logging file
util.check_dir(config.path_log)
util.set_logger(os.path.join(config.path_log, 'train.log'))

# Checking for param json folder and creating logging file
param_dict = config.model_param
util.save_json(param_dict, config.path_param)

logging.info('Loading Parameters')
params = util.Params(config.path_param)

# Training Data to Data Generator format
logging.info('Training Data to Data Generator format')
data = util.load_txt(config.path_training) + util.load_txt(config.path_testing)
index = np.arange(len(data))
np.random.shuffle(index)


# Validation Data to Data Generator format
logging.info('Preprocessin data')
data = ''.join(data)
data = pre_processing.preprocess_str(data)

# Converting again to list
data = data.split()

words = len(data)
vocab_size = params.vocab_size # Vocab size
data, count, dictionary, reversed_dictionary = pre_processing.build_dataset(data, vocab_size)

window_size = params.window_size
vector_dim = params.dimension_size
epochs = params.epochs

valid_size = params.valid_size     # Random set of words to evaluate similarity on.
valid_window = params.valid_window  # Only pick dev samples in the head of the distribution.
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

# Chosing word probability
sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

couples, labels = tf.keras.preprocessing.sequence.skipgrams(data, vocab_size, window_size=window_size, sampling_table=sampling_table)
word_target, word_context = zip(*couples)
word_target = np.array(word_target, dtype="int32")
word_context = np.array(word_context, dtype="int32")

n = len(couples)
train_x = couples[:int(0.8*n)]
test_x = couples[int(0.8*n):]

train_y = labels[:int(0.8*n)]
test_y = labels[int(0.8*n):]

# Initilizing Model
logging.info('Initializing Model')

# Model Training
path_save_callback = os.path.join(
    config.path_output_model, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
savingModel = tf.keras.callbacks.ModelCheckpoint(path_save_callback,
                                              monitor='val_loss',
                                              verbose=0,
                                              save_best_only=False,
                                              save_weights_only=True,
                                              mode='auto',
                                              period=5)

csvLogger = tf.keras.callbacks.CSVLogger(
    os.path.join(config.path_log, 'trainingLog.csv'),  append=True)

word2vec_model = model.word2vecModel(params)

logging.info('Compiling Model')
word2vec_model.compile(optimizer=tf.keras.optimizers.Adam(lr=params.learning_rate),
                       loss='binary_crossentropy')

word2vec_model.summary()
logging.info('Training Started...')
try:
    history = word2vec_model.fit(x=train_x, y=train_y, 
                                batch_size=params.batch_size,
                                validation_data=(test_x, test_y),
                                epochs=params.epochs,
                                # sample_weight=weight_vector,
                                use_multiprocessing=True,
                                workers=10,
                                callbacks=[savingModel, csvLogger]
                                )
except Exception as e:
    msg = 'Unable to start/complete model training. Error : {}'.format(e)
    logging.error(msg)
    raise(msg)

logging.info('Saving Last model...')
path_trained_model = os.path.join(
    config.path_output_model, 'retinaModel.h5')

word2vec_model.save(path_trained_model)
logging.info('word2vec Model: {}'.format(path_trained_model))
