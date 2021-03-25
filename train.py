import tensorflow as tf
import keras
import config
import os
import argparse
import logging

from src import model
from src import preprocessing
from src import dataGenerator
from src import util
from src import model


physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
_config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
get_training_data = dataGenerator.getData(path_data=config.path_training)
training_indexes, training_image_map, training_label_map = get_training_data.getList()

logging.info('Creating Training Data Generator')
trainingGenerator = dataGenerator.DataGenerator(
    list_IDs=training_indexes, imageMap=training_image_map, labelMap=training_label_map, params=params, prediction=False, shuffle=True)

# Validation Data to Data Generator format
logging.info('Validation data to Data Generator format')
getValidationData = dataGenerator.getData(path_data=config.path_testing)
validation_indexes, validation_image_map, validation_label_map = getValidationData.getList()

logging.info('Creating Training Data Generator')
validationGenerator = dataGenerator.DataGenerator(
    list_IDs=validation_indexes, imageMap=validation_image_map, labelMap=validation_label_map, params=params, prediction=False, shuffle=True)

# Initilizing Model
logging.info('Initializing Model')

# Model Training
path_save_callback = os.path.join(
    config.pathOutputModel, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
savingModel = keras.callbacks.ModelCheckpoint(path_save_callback,
                                              monitor='val_loss',
                                              verbose=0,
                                              save_best_only=False,
                                              save_weights_only=True,
                                              mode='auto',
                                              period=5)

csvLogger = keras.callbacks.CSVLogger(
    os.path.join(config.pathLog, 'trainingLog.csv'),  append=True)

retinaModel, retinaHeatMap = model.retinaModel(params)

logging.info('Compiling Model')
retinaModel.compile(optimizer=keras.optimizers.Adam(lr=params.learningRate),
                    loss='binary_crossentropy')

retinaModel.summary()
logging.info('Training Started...')
try:
    history = retinaModel.fit_generator(generator=trainingGenerator,
                                        validation_data=validationGenerator,
                                        epochs=50,
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
pathTrainedRetinaModel = os.path.join(
    config.pathOutputModel, 'retinaModel.h5')
pathTrainedHeatMapModel = os.path.join(
    config.pathOutputModel, 'heatMapModel.h5')

retinaModel.save(pathTrainedRetinaModel)
retinaHeatMap.save(pathTrainedHeatMapModel)
logging.info('Retina Model: {}, HeatMap Model: {}'.format(
    pathTrainedRetinaModel, pathTrainedHeatMapModel))
