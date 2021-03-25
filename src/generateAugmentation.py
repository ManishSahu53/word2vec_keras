import os
import config
from src import util

import numpy as np
import cv2
import os
from tqdm import tqdm
import logging

paramDict = config.modelParam
util.save_json(paramDict, config.pathParam)
params = util.Params(config.pathParam)

# Checking for logging folder and creating logging file
util.check_dir(config.pathLog)
util.set_logger(os.path.join(config.pathLog, 'generateAugmentation.log'))

logging.info(' Reading all the image file present in training data folder')
# Reading all the image file present in training data folder
pathTrainingData = []
for root, dirs, files in os.walk(config.pathTraining):
    for file in files:
        if file.endswith(tuple(config.imageFormat)):
            pathTrainingData.append(os.path.join(root, file))

# Checking for Image augmentations
xflip = params.xflip
yflip = params.yflip

if (xflip) and (yflip) is True:
    bothflip = True

logging.info('Doing Training Image augmentations')
# Doing Training Image augmentations
for temp_pathImage in tqdm(pathTrainingData):
    img = cv2.imread(temp_pathImage)
    if xflip:
        flipVertical = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(config.pathTraining, util.getNamenoExt(temp_pathImage) +
                    '_xflip.jpg'), flipVertical)

    if yflip:
        flipHorizontal = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(config.pathTraining, util.getNamenoExt(temp_pathImage) +
                            '_yflip.jpg'), flipHorizontal)

    if bothflip:
        flipBoth = cv2.flip(img, -1)
        cv2.imwrite(os.path.join(config.pathTraining, util.getNamenoExt(temp_pathImage) +
                    '_bflip.jpg'), flipBoth)

logging.info('Reading all the images present inside the testing folder')
# Reading all the images present inside the testing folder
pathTestingData = []
for root, dirs, files in os.walk(config.pathTesting):
    for file in files:
        if file.endswith(tuple(config.imageFormat)):
            pathTestingData.append(os.path.join(root, file))


logging.info('Doing Testing Image augmentations')
# Doing Testing Image augmentations
for temp_pathImage in tqdm(pathTestingData):
    img = cv2.imread(temp_pathImage)
    if xflip:
        flipVertical = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(config.pathTesting, util.getNamenoExt(temp_pathImage) +
                    '_xflip.jpg'), flipVertical)

    if yflip:
        flipHorizontal = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(config.pathTesting, util.getNamenoExt(temp_pathImage) +
                            '_yflip.jpg'), flipVertical)

    if bothflip:
        flipBoth = cv2.flip(img, -1)
        cv2.imwrite(os.path.join(config.pathTesting, util.getNamenoExt(temp_pathImage) +
                    '_bflip.jpg'), flipVertical)


logging.info('Completed')