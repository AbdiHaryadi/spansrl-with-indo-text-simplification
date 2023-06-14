import sys
import json
import numpy as np
import tensorflow as tf
from .helper import SRLData, split_train_test_val
from ..utils import print_default

def main():
    config = sys.argv[1]
    filename = './configurations.json'
    f = open(filename)
    all_config = json.load(f)

    try:
        config = all_config[config]
    except:
        print_default()
        config = all_config['default']
    
    srl_data = SRLData(config)

    # Read raw data to get sentences and argument list
    sent, arg_list = srl_data.read_raw_data()

    # Extract features from sentences
    srl_data.extract_features(sent)

    # Convert output
    out = srl_data.convert_train_output(arg_list)

    # Split train test val, and saving them into features directory
    features_1 = srl_data.word_emb_w2v
    features_11 = srl_data.word_emb_ft
    features_2 =  srl_data.word_emb_2
    features_3 = srl_data.char_input
    split_train_test_val(features_1, features_11, features_2, features_3, out, sent, config)

if __name__ == "__main__":
    with tf.device('/gpu:3'):
        main()