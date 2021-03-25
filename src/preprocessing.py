import os
import config
from src import util
import logging


# Get Labels from name of image
def getLabel(pathImage: str) -> int:

    """
        It generates label from image file name
        Input:
            path of image or name of image
        Output:
            int (1 - Good, 0 - Bad)
    """
    basename = util.getNamenoExt(pathImage)
    if 'bad' in basename:
        output = 0 # Bad

    elif 'good' in basename:
        output = 1 # Good
    
    else:
        logging.error('Error: No label found in the name')
        raise('Error: No label found in the name')
    
    return output