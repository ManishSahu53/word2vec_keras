import keras
import numpy as np
import logging
import cv2
import os

from src import preprocessing
import config


# Generating dataset
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, imageMap, labelMap, params,
                 prediction=False, shuffle=True):
        'Initialization'
        self.labelMap = labelMap
        self.list_IDs = list_IDs
        self.imageMap = imageMap
        self.prediction = prediction
        self.params = params
        self.shuffle = shuffle
        self.on_epoch_end()

        # While Prediction never shuffle the dataset
        if self.prediction is True:
            self.shuffle = False

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.params.batchSize))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.params.batchSize:(index+1)*self.params.batchSize]

        # Capping this the data size, else will be multiple of batch size
        indexes = indexes[:len(self.list_IDs)]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if self.prediction is False:
            # Generate data
            X, y = self.__data_generation(list_IDs_temp)
            return X, y

        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'

        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def getLabel(self, id: str):
        label = preprocessing.getLabel(self.labelMap[id])

        logging.info('label: {}'.format(label))
        return label
    
    def getImage(self, id: str):
        try:
            img = cv2.imread(self.imageMap[id])
            img = cv2.resize(img, (self.params.imageSize, self.params.imageSize))
            if self.params.normalize:
                img = img/255.0

        except Exception as e:
            logging.error(
                'Unable to read image: {}. Error: {}'.format(self.imageMap[id], e))
            img = np.zeros([self.params.imageSize, self.params.imageSize, self.params.imageDim])

        return img


    def __data_generation(self, list_IDs_temp):
        # X : (n_samples, *dim, n_channels)
        'Generates data containing batchSize samples'

        # Initialization
        word1 = np.zeros(
            (self.params.batch_size, self.params.dimension_size, self.params.max_subword), dtype=float)
        
        word2 = np.zeros(
            (self.params.batch_size, self.params.dimension_size, self.params.max_subword), dtype=float)
        
        y = np.zeros(
            (self.params.batch_size, self.params.max_subword, 1), dtype=int)

        # Generating data

        # If Training
        if self.prediction is False:
            for i, ID in enumerate(list_IDs_temp):

                # Extracting image and Label values image file
                X[i] = self.getImage(ID)
                y[i] = self.labelMap[ID]

            return X, y

        # If Predicting id_vector
        else:
            for i, ID in enumerate(list_IDs_temp):
                X[i] = self.getImage(ID)
            return X


# Generating Training data map
class getData():
    """
        Convert path director of image into data generator format

    """

    def __init__(self, path_data):
        self.path_data = path_data


    def getList(self):
        """
        Convert path director of image into data generator format

        Output: 
            key: indexes
            imageMap: indexes and path image map
            labelMap: indexes and path label map

        """
        labelMap = {}
        imageMap = {}
        key = []
        index = 0

        for root, dirs, files in os.walk(self.path_data):
            for file in files:
                # If .png or .jpg file found then
                if file.endswith(tuple(config.imageFormat)):
                    key.append(index)
                    labelMap[index] = preprocessing.getLabel(file)
                    imageMap[index] = os.path.join(root, file)

                    index += 1

                else:
                    continue

        return key, imageMap, labelMap
