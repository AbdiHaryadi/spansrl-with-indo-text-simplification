import sys
import json

import tensorflow as tf
from .utils import eval_test

config = sys.argv[1]
filename = './configurations.json'
f = open(filename)
all_config = json.load(f)
config = all_config[config]

print('Validating test data with model ' + config['model_path'])

with tf.device('/gpu:0'):
    eval_test(config)
