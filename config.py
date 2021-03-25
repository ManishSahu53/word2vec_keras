import os
from src import util

experiment = 1

path_training = 'data/football/events.csv'
path_output = 'experiments/e1'
path_testing = 'data/test'
path_log = 'log'

model_param = {
    'dimension_size': 100,
    'max_subword': 1,
    'batch_size': 1024,
    'learning_rate' : 10e-4,

}


####################### DO NOT CHANGE FROM HERE #########################
path_param = os.path.join(path_output, 'param.json')
path_output_model = os.path.join(path_output, 'model')

util.check_dir(path_output)
util.check_dir(path_log)
util.check_dir(path_output_model)


