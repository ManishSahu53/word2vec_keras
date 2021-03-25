import os
from src import util

experiment = 1

path_training = 'data/train.txt'
path_testing = 'data/test.txt'

path_output = 'experiments/e1'
path_log = 'log'

model_param = {
    'dimension_size': 100,
    'max_subword': 1,
    'batch_size': 1024,
    'learning_rate' : 10e-4,
    'window_size' :  3,
    'epochs' : 20000,
    'valid_size' : 16,     # Random set of words to evaluate similarity on.
    'valid_window' : 100,  # Only pick dev samples in the head of the distribution.
    'vocab_size': 5000

}


####################### DO NOT CHANGE FROM HERE #########################
path_param = os.path.join(path_output, 'param.json')
path_output_model = os.path.join(path_output, 'model')

util.check_dir(path_output)
util.check_dir(path_log)
util.check_dir(path_output_model)


